from typing import Dict, Any, List
import numpy as np
import tensorflow as tf
# Tensorleap imports
from code_loader import leap_binder
from code_loader.contract.enums import LeapDataType
from code_loader.contract.datasetclasses import PreprocessResponse
from code_loader.inner_leap_binder.leapbinder_decorators import (tensorleap_preprocess, tensorleap_unlabeled_preprocess,
                                                                 tensorleap_input_encoder, tensorleap_gt_encoder)

from NER.dataset import load_data, downsample_hf_dataset
from NER.config import CONFIG
from NER.ner import tokenize_and_align_labels, tokenizer
from tl.visualizers import input_visualizer, text_visualizer_mask_gt, text_visualizer_mask_pred, text_visualizer_mask_comb
from tl.metadata_helpers import metadata_dic
from NER.utils.metrics import compute_entity_entropy_per_sample, count_splitting_merging_errors, calc_metrics

import subprocess
import sys

def install(package, version=None):
    if version:
        package_name = f"{package}=={version}"
    else:
        package_name = package
    subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])

install("numpy", "1.24.4")  # Replace "1.21.0" with your desired version
install("joblib", "1.4.2")  # Replace "1.21.0" with your desired version



def decode_text(i: int, subset: PreprocessResponse):
    tokens = subset.data['ds'][i]
    tokenized_inputs = tokenizer(tokens['tokens'], truncation=True, is_split_into_words=True)['input_ids']
    text_data = tokenizer.decode(tokenized_inputs, skip_special_tokens=True)
    return text_data



@tensorleap_preprocess()
def preprocess_func() -> List[PreprocessResponse]:

    ds_train, ds_val, ds_test = load_data()

    ds_train = downsample_hf_dataset(ds_train, CONFIG["train_size"])
    ds_val = downsample_hf_dataset(ds_val, CONFIG["val_size"])

    # take first half for test subset
    idx = len(ds_test)//2
    ds_test = ds_test.select(np.arange(idx))
    ds_test = downsample_hf_dataset(ds_test, CONFIG["test_size"])

    # Generate a PreprocessResponse for each data slice, to later be read by the encoders.
    # The length of each data slice is provided, along with the data dictionary.
    # In this example we pass `images` and `labels` that later are encoded into the inputs and outputs 
    train = PreprocessResponse(length=len(ds_train), data={'ds': ds_train, 'subset': 'train'})
    val = PreprocessResponse(length=len(ds_val), data={'ds': ds_val, 'subset': 'val'})
    test = PreprocessResponse(length=len(ds_test), data={'ds': ds_test, 'subset': 'test'})
    response = [train, val, test]
    return response


@tensorleap_unlabeled_preprocess()
def preprocess_func_ul() -> List[PreprocessResponse]:
    _, _, ds_test = load_data()

    # take second half for unlabeled subset
    idx = len(ds_test)//2
    ds_test = ds_test.select(np.arange(idx, len(ds_test)))
    ds_test = downsample_hf_dataset(ds_test, CONFIG["ul_size"])

    # Generate a PreprocessResponse for each data slice, to later be read by the encoders.
    # The length of each data slice is provided, along with the data dictionary.
    # In this example we pass `images` and `labels` that later are encoded into the inputs and outputs
    response = PreprocessResponse(length=len(ds_test), data={'ds': ds_test, 'subset': 'ul'})
    return response

# Input encoder fetches the image with the index `idx` from the `images` array set in
# the PreprocessResponse data. Returns a numpy array containing the sample's image. 


def input_encoder(idx: int, preprocess: PreprocessResponse) -> np.ndarray:
    sample = preprocess.data['ds'][idx:idx+1] #['tokens']
    tokenized_inputs = tokenize_and_align_labels(sample)
    return tokenized_inputs


@tensorleap_input_encoder(name="input_ids")
def input_ids(idx: int, preprocess: PreprocessResponse) -> np.ndarray:
    inputs = input_encoder(idx, preprocess).data
    inputs = inputs["input_ids"][0]
    return inputs.numpy().astype(np.float32)


@tensorleap_input_encoder(name="input_type_ids")
def input_type_ids(idx: int, preprocess: PreprocessResponse) -> np.ndarray:
    inputs = input_encoder(idx, preprocess).data
    inputs = inputs["token_type_ids"][0]
    return inputs.numpy().astype(np.float32)


@tensorleap_input_encoder(name="attention_mask")
def input_attention_mask(idx: int, preprocess: PreprocessResponse) -> np.ndarray:
    inputs = input_encoder(idx, preprocess).data
    inputs = inputs["attention_mask"][0]
    return inputs.numpy().astype(np.float32)



# Ground truth encoder fetches the label with the index `idx` from the `labels` array set in
# the PreprocessResponse's data. Returns a numpy array containing a hot vector label correlated with the sample.
@tensorleap_gt_encoder(name="attention_mask")
def gt_encoder(idx: int, preprocess: PreprocessResponse) -> np.ndarray:
    tokenized_inputs = input_encoder(idx, preprocess)   # get tokenized labels
    labels = tokenized_inputs.data["labels"]
    gt_tensor_one_hot = tf.one_hot(labels, depth=len(CONFIG["labels"]))
    return gt_tensor_one_hot[0].numpy()




if __name__ == "__main__":
    leap_binder.check()
