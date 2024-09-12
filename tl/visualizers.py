from typing import Union, Any, Dict

# Tensorleap imports
from code_loader.contract.visualizer_classes import LeapText, LeapTextMask


from NER.ner import *
from NER.utils.ner import *
from tl.metadata_helpers import *

# TODO: fix take fron the mask vis
def class_visualizer(tensor: np.ndarray) -> LeapText:
    _, text = decode(tensor)
    text = truncate_pad(text[0], CONFIG["labels"][0])
    return LeapText(text)


def input_visualizer(input_ids: np.ndarray) -> LeapText:
    input_ids = input_ids[0] if len(input_ids.shape) == 2 else input_ids    # flatten for batch shape
    input_ids = tf.cast(input_ids, dtype=tf.int32)
    text = decode_token_ids(input_ids)
    return LeapText(text)


def text_visualizer_mask(input_ids: np.ndarray, gt_vec_labels: Union[tf.Tensor, np.ndarray], pred_vec_labels: Union[tf.Tensor, np.ndarray] = None) -> LeapTextMask:

    gt_vec_labels = gt_vec_labels.numpy() if isinstance(gt_vec_labels, tf.Tensor) else gt_vec_labels  # convert to numpy
    gt_vec_labels = gt_vec_labels[None, ...].argmax(-1)  # from one-hot to labels

    if pred_vec_labels is not None:   # Vis Pred
        _, pred_vec_labels = postprocess_predictions(pred_vec_labels[None, ...], gt_vec_labels)
        labels_names = pred_vec_labels[0]    # get single sample
    else:           # Vis GT
        gt_vec_labels = postprocess_labels(gt_vec_labels)
        labels_names = gt_vec_labels[0]  # get single sample

    cat_to_int = {c: i for i, c in enumerate(CONFIG["categories"])}
    # Decode token IDS to text tokens in list
    text_tokens = decode_token_ids(input_ids)

    # take the category of each
    labels_names = [c.split("-")[-1] for c in labels_names]
    mask = np.array([cat_to_int[c] for c in labels_names]).astype(np.uint8)

    return LeapTextMask(text=text_tokens, mask=mask, labels=CONFIG["categories"])

