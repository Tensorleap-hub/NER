from typing import Union, Any, Dict

# Tensorleap imports
from code_loader import leap_binder
from code_loader.contract.datasetclasses import PreprocessResponse
from code_loader.contract.visualizer_classes import LeapText, LeapTextMask
from code_loader.contract.enums import LeapDataType


from NER.dataset import load_data, downsample_hf_dataset
from NER.utils.metrics import *
from tl.metadata_helpers import *
from tl.visualizers import *



# Preprocess Function
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

def input_ids(idx: int, preprocess: PreprocessResponse) -> np.ndarray:
    inputs = input_encoder(idx, preprocess).data
    inputs = inputs["input_ids"][0]
    return inputs

def input_type_ids(idx: int, preprocess: PreprocessResponse) -> np.ndarray:
    inputs = input_encoder(idx, preprocess).data
    inputs = inputs["token_type_ids"][0]
    return inputs

def input_attention_mask(idx: int, preprocess: PreprocessResponse) -> np.ndarray:
    inputs = input_encoder(idx, preprocess).data
    inputs = inputs["attention_mask"][0]
    return inputs



# Ground truth encoder fetches the label with the index `idx` from the `labels` array set in
# the PreprocessResponse's data. Returns a numpy array containing a hot vector label correlated with the sample.
def gt_encoder(idx: int, preprocess: PreprocessResponse) -> np.ndarray:
    tokenized_inputs = input_encoder(idx, preprocess)   # get tokenized labels
    labels = tokenized_inputs.data["labels"]
    gt_tensor_one_hot = tf.one_hot(labels, depth=len(CONFIG["labels"]))
    return gt_tensor_one_hot[0]



def metadata_language(idx: int, preprocess: PreprocessResponse) -> int:
    id_text = preprocess.data['ds'][idx]['id']
    language_code = id_text.split('-')[0]
    # TODO: add also language code integer
    return language_code


def metadata_text(idx: int, preprocess: PreprocessResponse) -> Dict[str, Any]:
    res = {}
    tokens = input_encoder(idx, preprocess).tokens()
    res['tokens_length'] = len(tokens)
    # TODO: add also language code integer
    return res




# Dataset binding functions to bind the functions above to the `Dataset Instance`.

# Data TL Preprocess Response
leap_binder.set_preprocess(function=preprocess_func)
leap_binder.set_unlabeled_data_preprocess(function=preprocess_func_ul)

# Input and GT Preprocess
leap_binder.set_input(function=input_ids, name='input_ids')
leap_binder.set_input(function=input_type_ids, name='input_type_ids')
leap_binder.set_input(function=input_attention_mask, name='attention_mask')
leap_binder.set_ground_truth(function=gt_encoder, name='ner_tags')

# Metadata variables
leap_binder.set_metadata(function=metadata_dic, name='metadata_dic')

# Loss
leap_binder.add_custom_loss(function=CE_loss, name="CE_loss")

# Metrics
leap_binder.add_custom_metric(function=calc_metrics, name="metrics")
leap_binder.add_custom_metric(function=compute_entity_entropy_per_sample, name="avg_entity_entropy")
leap_binder.add_custom_metric(function=count_splitting_merging_errors, name="errors")

# The Prediction Labels
leap_binder.add_prediction(name='classes', labels=CONFIG["model_labels"], channel_dim=1)

# TL Visualizers
# leap_binder.set_visualizer(function=raw_text_visualizer, visualizer_type=LeapDataType.Text, name="raw_text_visualizer")
leap_binder.set_visualizer(function=input_visualizer, visualizer_type=LeapDataType.Text, name="input_visualizer")
leap_binder.set_visualizer(function=text_visualizer_mask_gt, visualizer_type=LeapDataType.TextMask, name="mask_visualizer_gt")
leap_binder.set_visualizer(function=text_visualizer_mask_pred, visualizer_type=LeapDataType.TextMask, name="mask_visualizer_pred")
leap_binder.set_visualizer(function=text_visualizer_mask_comb, visualizer_type=LeapDataType.TextMask, name="mask_visualizer_comb")
leap_binder.set_visualizer(function=loss_visualizer, visualizer_type=LeapDataType.Image, name="loss_visualizer")

if __name__ == "__main__":
    leap_binder.check()
