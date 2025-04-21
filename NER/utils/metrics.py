import numpy as np
from NER.config import CONFIG
import tensorflow as tf
from code_loader.inner_leap_binder.leapbinder_decorators import tensorleap_custom_metric

from NER.utils.ner import transform_prediction
from NER.ner import mask_one_hot_labels, map_label_idx_to_cat, map_idx_to_label


@tensorleap_custom_metric(name="metrics")
def calc_metrics(ground_truth: np.ndarray, prediction: np.ndarray):
    """`
    Calculate Accuracy, Precision, Recall, and F1 Score for NER.
    - Total metric score
    - Per each category class

    Parameters:
    true_labels (list of lists): True labels for each token.
    predicted_labels (list of lists): Predicted labels for each token.

    Returns:
    dictionary with:
        precision (float): Precision score
        recall (float): Recall score
        f1_score (float): F1 score

        [C]_precision, [C]_recall, [C]_f1_score for any category class C: [LOC, ORG, PER, MISC]

    """
    ground_truth = tf.convert_to_tensor(ground_truth)
    prediction = tf.convert_to_tensor(prediction)

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
        """ Calc the detailed metrics in sample level"""

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

    # Calc per each sample in batch
    for sample_ground_truth, sample_prediction, sample_mask in zip(ground_truth, prediction, batch_mask):
        dic_scores = sample_precision_recall_f1(sample_ground_truth, sample_prediction, sample_mask)
        for k in metrics:
            metrics[k].append(dic_scores[k])

    # Convert to tensorflow tensor
    for k, v in metrics.items():
        metrics[k] = np.array(v, dtype=np.float32)

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

@tensorleap_custom_metric(name="avg_entity_entropy")
def compute_entity_entropy_per_sample(ground_truth: np.ndarray, prediction: np.ndarray):
    """
    Compute the entropy for entities only in each sample.

    :param prob_distributions: List of probability distributions, one per token.
    :param entity_labels: List of labels for each token (e.g., 'B-ORG', 'I-PER').
    :return: Single entropy score per sample, using the mean of entity token entropies.
    """

    ground_truth = tf.convert_to_tensor(ground_truth)
    prediction = tf.convert_to_tensor(prediction)

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
    return (tf.reduce_mean(entropies, -1) if entropies else tf.zeros_like(ground_truth)).numpy() # Handle case with no entities


def extract_entities(label_sequence):
    """
    Extracts a list of entities from a sequence of BIO-tagged labels.

    Parameters:
    label_sequence (list of str): A list of labels in BIO (Begin, Inside, Outside) format.
        e.g., ['B-PER', 'I-PER', 'O', 'B-LOC', 'I-LOC']

    Returns:
    list of list of tuples: A list where each element is a list of tuples, each tuple representing
    an entity. Each tuple contains the index of the label in the original list and the entity type.
    Each list of tuples represents all tokens that belong to a single entity.

    Example:
    />>> extract_entities(['B-PER', 'I-PER', 'O', 'B-LOC', 'I-LOC'])
    [[(0, 'PER'), (1, 'PER')], [(3, 'LOC'), (4, 'LOC')]]

    The function processes the label sequence, identifying entities based on 'B-' (begin) and 'I-'
    (inside) tags. It groups contiguous 'B-' and 'I-' tags as single entities and handles 'O' tags as
    separators between entities.
    """
    entities = []
    current_entity = []
    for index, label in enumerate(label_sequence):
        if label.startswith('B-'):      # start new entity instance
            if current_entity:
                entities.append(current_entity)
                current_entity = []
            current_entity.append((index, label[2:]))
        elif label.startswith('I-') and current_entity:     # inside given entity instance
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


@tensorleap_custom_metric(name="errors")
def count_splitting_merging_errors(ground_truth: np.ndarray, prediction: np.ndarray):
    """
    Calculates the number of splitting and merging errors in named entity recognition predictions compared to the ground truth.

    Parameters:
    - ground_truth (np.ndarray): A np.ndarray of shape (batch_size, sequence_length, num_labels) containing the one-hot encoded labels of the ground truth.
    - prediction (np.ndarray): A np.ndarray of shape (batch_size, sequence_length, num_labels) containing the model predictions, which may be logits or probabilities that need to be transformed into label indices.

    Returns:
    - dict: A dictionary containing two tensors 'splitting_errors' and 'merging_errors' each of shape (batch_size,). These tensors count the number of splitting and merging errors for each sample in the batch.

    This function masks irrelevant labels, aligns the predictions with the ground truth, and converts the model outputs to discrete labels.
    It then extracts named entities using a BIO tagging scheme and compares the ground truth and prediction spans to determine the count of splitting and merging errors:
    - Splitting errors occur when a single ground truth entity is split into multiple entities in the predictions.
    - Merging errors occur when multiple ground truth entities are merged into a single entity in the predictions.

    """
    ground_truth = tf.convert_to_tensor(ground_truth)
    prediction = tf.convert_to_tensor(prediction)

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

        scores["splitting_errors"].append(splitting_errors)
        scores["merging_errors"].append(merging_errors)

    # # Convert to tensors
    scores["splitting_errors"] = np.array(scores["splitting_errors"], dtype=np.float32)
    scores["merging_errors"] = np.array(scores["merging_errors"], dtype=np.float32)
    return scores
