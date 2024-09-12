from typing import List, Tuple
from NER.config import CONFIG
import tensorflow as tf


def hf_get_labels(ds) -> List[str]:
    """ Given HF dataset, return task labels """
    ner_feature = ds.features["ner_tags"]
    label_names = ner_feature.feature.names
    return label_names


def hf_decode_labels(sample) -> Tuple[str, str]:
    """ Given HF dataset sample, return decoded text and corresponding labels as strings """
    label_names = CONFIG["labels"]
    words = sample["tokens"]
    labels = sample["ner_tags"]
    line1 = ""
    line2 = ""
    for word, label in zip(words, labels):
        full_label = label_names[label]
        max_length = max(len(word), len(full_label))
        line1 += word + " " * (max_length - len(word) + 1)
        line2 += full_label + " " * (max_length - len(full_label) + 1)

    return line1, line2

def _tag_to_entity_type(tag: str) -> str:
    return tag.split("-")[-1]


def _is_entity(tag: str) -> bool:
    return tag != CONFIG["labels"][0]

def transform_prediction(tensor: tf.Tensor):
    """ Check if need to transform the tensor """
    if tensor.shape[-1] != len(CONFIG['labels']):
        tensor = tf.transpose(tensor, perm=[0, 2, 1])
    return tensor


def align_labels_with_tokens(labels, word_ids):
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            # Start of a new word!
            current_word = word_id
            label = -100 if word_id is None else labels[word_id]
            new_labels.append(label)
        elif word_id is None:
            # Special token
            new_labels.append(-100)
        else:
            # Same word as previous token
            label = labels[word_id]
            # If the label is B-XXX we change it to I-XXX
            if label % 2 == 1:
                label += 1
            new_labels.append(label)

    return new_labels

