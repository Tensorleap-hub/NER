import os

from code_loader.inner_leap_binder.leapbinder_decorators import tensorleap_load_model, integration_test
from code_loader.plot_functions.visualize import visualize

from leap_binder import preprocess_func, preprocess_func_ul, input_encoder, gt_encoder
import tensorflow as tf

from tqdm import tqdm
from transformers import TFAutoModelForTokenClassification

from NER.ner import *
from leap_binder import *

@tensorleap_load_model([])
def load_model():
    H5_MODEL_PATH = "NER/NER/model/ner.h5"
    dir_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(dir_path, H5_MODEL_PATH)
    return tf.keras.models.load_model(os.path.join(dir_path, model_path))

@integration_test()
def check_custom_integration(idx, responses_set):
    idx = 425

    print("Starting custom tests")
    model = load_model()

    gt = gt_encoder(idx, responses_set)
    input1 = input_ids(idx, responses_set)
    input2 = input_attention_mask(idx, responses_set)
    input3 = input_type_ids(idx, responses_set)

    pred = model([input1, input2, input3])

    vis1 = input_visualizer(input1)
    vis2 = text_visualizer_mask_gt(input1, gt)
    vis3 = text_visualizer_mask_pred(input1, pred)
    vis4 = text_visualizer_mask_comb(input1, gt, pred)

    scores = count_splitting_merging_errors(gt, pred)
    scores = calc_metrics(gt, pred)
    res = compute_entity_entropy_per_sample(gt, pred)
    loss = CE_loss(gt, pred)

    visualize(vis1)
    visualize(vis2)
    visualize(vis3)
    visualize(vis4)


    res = metadata_dic(idx, responses_set)






if __name__ == '__main__':
    responses = preprocess_func()
    train = responses[2]
    check_custom_integration(0, train)