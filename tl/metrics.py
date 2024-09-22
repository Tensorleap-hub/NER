from typing import List, Tuple

import numpy as np
import tensorflow as tf

from code_loader.contract.datasetclasses import ConfusionMatrixElement
from code_loader.contract.enums import ConfusionMatrixValue

from NER.config import CONFIG
from NER.utils.metrics import mask_one_hot_labels, transform_prediction
from NER.ner import map_label_idx_to_cat


#
# def confusion_matrix_tl(ground_truth: tf.Tensor, prediction: tf.Tensor):
#     """`
#     Calculate CM elemnts for TL.
#
#     Parameters:
#     true_labels (list of lists): True labels for each token.
#     predicted_labels (list of lists): Predicted labels for each token.
#
#     Returns:
#     precision (float): Precision score
#     recall (float): Recall score
#     f1_score (float): F1 score
#     """
#     # TODO: add ignoring CLS PAD tokens etc
#     O_token = CONFIG["labels"][0]
#     categories = CONFIG["categories"][0]
#
#     # Get mask based on padded tokens
#     batch_mask = mask_one_hot_labels(ground_truth)
#     # Transform one-hot to labels
#     ground_truth = tf.argmax(ground_truth, -1)
#     # Transform the labels to map the GT labels
#     prediction = transform_prediction(prediction)
#
#     def sample_cm(sample_ground_truth, sample_prediction, sample_mask=None):
#
#         if sample_mask is not None:
#             assert len(sample_ground_truth) == len(sample_prediction) == len(
#                 sample_mask), "Mismatched number of sequences"
#         else:
#             assert len(sample_ground_truth) == len(sample_prediction), "Mismatched number of sequences"
#
#         # mask by given mask label
#         if sample_mask is not None:
#             sample_ground_truth = tf.boolean_mask(sample_ground_truth, sample_mask)
#             sample_prediction = tf.boolean_mask(sample_prediction, sample_mask)
#
#         cm_elems = []
#
#         for true_label, pred in zip(sample_ground_truth, sample_prediction):
#
#             true_label, pred = true_label.numpy(), pred.numpy()
#             pred_label = pred.argmax(-1)
#             prob = pred[pred_label]
#             cat = map_label_idx_to_cat[true_label]
#
#             if cat == O_token:      # Don't count if it's not an entity (TODO *?, do we want that?)
#                 continue
#
#             for c in categories[1:]:
#                 if pred_label == true_label and cat == c:  # TP
#                     elem = ConfusionMatrixElement(
#                         label=cat,
#                         expected_outcome=ConfusionMatrixValue.Positive,
#                         predicted_probability=float(prob),
#                     )
#                 elif pred_label == 0:       # FN
#                     elem = ConfusionMatrixElement(
#                         label=cat,
#                         expected_outcome=ConfusionMatrixValue.Positive,
#                         predicted_probability=float(prob),
#                     )
#                 else:
#
#                 cm_elems.append(elem)
#
#
#
