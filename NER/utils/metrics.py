from typing import List, Optional
import numpy as np
from NER.config import CONFIG
import tensorflow as tf
from transformers import AutoTokenizer

from NER.utils.ner import transform_prediction, align_labels_with_tokens
from NER.ner import mask_one_hot_labels, map_label_idx_to_cat, map_idx_to_label



def calc_metrics(ground_truth: tf.Tensor, prediction: tf.Tensor):
    """`
    Calculate Accuracy, Precision, Recall, and F1 Score for NER.
    - Total metric score
    - Per each category class

    Parameters:
    true_labels (list of lists): True labels for each token.
    predicted_labels (list of lists): Predicted labels for each token.

    Returns:
    precision (float): Precision score
    recall (float): Recall score
    f1_score (float): F1 score
    """
    # TODO: add ignoring CLS PAD tokens etc
    O_token = CONFIG["labels"][0]
    # Mask -100 labels
    batch_mask = mask_one_hot_labels(ground_truth)
    # Transform the prediction and swap the labels order to gt
    prediction = transform_prediction(prediction)
    # Transform to labels
    ground_truth = tf.argmax(ground_truth, -1)
    prediction = tf.argmax(prediction, -1)

    categories = list(map_label_idx_to_cat.values())
    metrics_names = ["accuracy", "precision", "recall", "f1_score"]
    metrics_class_names = [f"{cls}_{metric}" for cls in categories for metric in metrics_names]

    metrics_names.extend(metrics_class_names)

    metrics = {k: [] for k in metrics_names}

    def sample_precision_recall_f1(sample_ground_truth, sample_prediction, sample_mask=None):

        if sample_mask is not None:
            assert len(sample_ground_truth) == len(sample_prediction) == len(sample_mask), "Mismatched number of sequences"
        else:
            assert len(sample_ground_truth) == len(sample_prediction), "Mismatched number of sequences"

        # mask by -100 labels
        if sample_mask is not None:
            sample_ground_truth = tf.boolean_mask(sample_ground_truth, sample_mask)
            sample_prediction = tf.boolean_mask(sample_prediction, sample_mask)

        true_positives = 0
        false_positives = 0
        false_negatives = 0

        class_metrics = {}

        for true_label, pred_label in zip(sample_ground_truth, sample_prediction):
            true_label, pred_label = true_label.numpy(), pred_label.numpy()

            # Global All Classes TP, FN, ...
            if pred_label == true_label and pred_label != O_token:  # Count only entity labels
                true_positives += 1
            elif pred_label != true_label:
                if pred_label != O_token:
                    false_positives += 1
                if true_label != O_token:
                    false_negatives += 1

        sample_ground_truth, sample_prediction = np.array([map_label_idx_to_cat[l] for l in sample_ground_truth.numpy()]), np.array([map_label_idx_to_cat[l] for l in sample_prediction.numpy()])
        # Category-specific TP, FP, FN
        for cls in categories:
            TP = np.sum((sample_ground_truth == cls) & (sample_prediction == cls))
            FP = np.sum((sample_ground_truth != cls) & (sample_prediction == cls))
            FN = np.sum((sample_ground_truth == cls) & (sample_prediction != cls))
            total_class_samples = np.sum(sample_ground_truth == cls)

            # Calculate precision, recall, f1-score
            precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            accuracy = TP / total_class_samples if total_class_samples > 0 else 0

            # Assign class-specific metrics to the dictionary
            class_metrics.update({f'{cls}_precision': precision, f'{cls}_recall': recall, f'{cls}_f1_score': f1_score, f'{cls}_accuracy': accuracy})

        acc = tf.reduce_sum(tf.cast(sample_ground_truth == sample_prediction, tf.int32)) / max(1, len(sample_ground_truth))
        acc = acc.numpy().astype(np.float32)
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        return {"accuracy": acc, "precision": precision, "recall": recall, "f1_score": f1_score,
                **class_metrics}

    for sample_ground_truth, sample_prediction, sample_mask in zip(ground_truth, prediction, batch_mask):
        dic_scores = sample_precision_recall_f1(sample_ground_truth, sample_prediction, sample_mask)
        for k in metrics:
            metrics[k].append(dic_scores[k])

    for k, v in metrics.items():
        metrics[k] = tf.constant(v)

    return metrics


def shannon_entropy(prob_dist):
    """
    Compute Shannon entropy of a probability distribution.

    :param prob_dist: List or array of probabilities.
    :return: Shannon entropy of the distribution.
    """
    prob_dist = np.array(prob_dist)
    prob_dist = prob_dist[prob_dist > 0]
    return -np.sum(prob_dist * np.log(prob_dist))


def compute_entity_entropy_per_sample(ground_truth: tf.Tensor, prediction: tf.Tensor):
    """
    Compute the entropy for entities only in each sample.

    :param prob_distributions: List of probability distributions, one per token.
    :param entity_labels: List of labels for each token (e.g., 'B-ORG', 'I-PER').
    :return: Single entropy score per sample, using the mean of entity token entropies.
    """

    # TODO: add ignoring CLS PAD tokens etc
    # Transform map labels to GT
    prediction = transform_prediction(prediction)
    # Apply Softmax to the logits
    prediction = tf.nn.softmax(prediction, 1)

    # Convert one hot to labels
    ground_truth = tf.argmax(ground_truth, -1)

    # Filter probability distributions and labels to include only entities
    entity_prob_distributions = [[prob_dist for prob_dist, label in zip(pred, gt) if label > 0] for pred, gt in zip(prediction, ground_truth)]
    # Compute entropy for entity tokens only
    entropies = [[shannon_entropy(token_dist) for token_dist in dist] for dist in entity_prob_distributions]

    # Return the mean entropy of entity tokens, or another aggregation method
    return tf.reduce_mean(entropies, -1) if entropies else tf.zeros_like(ground_truth)  # Handle case with no entities


def extract_entities(label_sequence):
    """Extracts a list of entities from a sequence of BIO-tagged labels."""
    entities = []
    current_entity = []
    for index, label in enumerate(label_sequence):
        if label.startswith('B-'):
            if current_entity:
                entities.append(current_entity)
                current_entity = []
            current_entity.append((index, label[2:]))
        elif label.startswith('I-') and current_entity:
            current_entity.append((index, label[2:]))
        else:
            if current_entity:
                entities.append(current_entity)
                current_entity = []
    if current_entity:
        entities.append(current_entity)
    return entities


def count_splitted_intervals(inter_spans: dict, inter_withins: dict):
    overlapped = 0
    matching_inter_spans = 0
    i = 0
    inter_within_lst = list(inter_withins.keys())
    for inter_span, inter_type in inter_spans.items():
        matching_inter_spans = 0

        while i < len(inter_within_lst):
            inter_within = inter_within_lst[i]

            if inter_span[0] <= inter_within[0] <= inter_span[1]:
                matching_inter_spans += 1
            elif inter_within[0] > inter_span[1]:   # no match, break to next gt entity
                break
            i += 1  # move pointer

        if matching_inter_spans > 1:
            overlapped += 1
    return overlapped



def count_splitting_merging_errors(ground_truth: tf.Tensor, prediction: tf.Tensor):
    # Mask irrelevant labels
    batch_mask = mask_one_hot_labels(ground_truth)
    # Transform the prediction and swap the labels order according to gt
    prediction = transform_prediction(prediction)
    # Transform logits to labels
    ground_truth = tf.argmax(ground_truth, -1)
    prediction = tf.argmax(prediction, -1)

    ground_truth = ground_truth.numpy()
    prediction = prediction.numpy()

    # Mask Gt and Pred accordingly
    ground_truth = np.stack([gt[mask] for gt, mask in zip(ground_truth, batch_mask)])
    prediction = np.stack([pred[mask] for pred, mask in zip(prediction, batch_mask)])
    # To label names
    gt_labels = [[map_idx_to_label[i] for i in gt] for gt in ground_truth]
    pred_labels = [[map_idx_to_label[i] for i in pred] for pred in prediction]
    # Extract the separated entities
    gt_entities = [extract_entities(labels) for labels in gt_labels]
    pred_entities = [extract_entities(labels) for labels in pred_labels]


    gt_spans = [{(ent[0][0], ent[-1][0]): ent[0][1] for ent in sample} for sample in gt_entities]
    pred_spans = [{(ent[0][0], ent[-1][0]): ent[0][1] for ent in sample} for sample in pred_entities]

    scores = {"splitting_errors": [], "merging_errors": []}

    for gt_sample_spans, pred_sample_spans in zip(gt_spans, pred_spans):
        # Check for splitting errors
        splitting_errors = count_splitted_intervals(inter_spans=gt_sample_spans, inter_withins=pred_sample_spans)
        # Check for merging errors
        merging_errors = count_splitted_intervals(inter_spans=pred_sample_spans, inter_withins=gt_sample_spans)


        # for gt_span, gt_type in gt_sample_spans.items():
        #     matching_pred_spans = [pred_span for pred_span in pred_sample_spans if
        #                            pred_span[0] >= gt_span[0]]
        #     if len(matching_pred_spans) > 1:
        #         splitting_errors += 1
        #
        # for pred_span, pred_type in pred_sample_spans.items():
        #     matching_gt_spans = [gt_span for gt_span in gt_spans if
        #                          gt_span[0] >= pred_span[0] and gt_span[1] <= pred_span[1]]
        #     if len(matching_gt_spans) > 1:
        #         merging_errors += 1


        scores["splitting_errors"].append(splitting_errors)
        scores["merging_errors"].append(merging_errors)

    # Convert to tensors
    scores["splitting_errors"] = tf.constant(scores["splitting_errors"])
    scores["merging_errors"] = tf.constant(scores["merging_errors"])
    return scores
