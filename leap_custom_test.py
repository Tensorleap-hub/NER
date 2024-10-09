import os
import tensorflow as tf
from tqdm import tqdm
from transformers import TFAutoModelForTokenClassification

from code_loader.helpers import visualize

from NER.ner import *
from leap_binder import preprocess_func, preprocess_func_ul, input_encoder, gt_encoder, input_ids, input_attention_mask, input_type_ids
from tl.metadata_helpers import metadata_dic, get_sample_topic
from tl.visualizers import input_visualizer, text_visualizer_mask_comb, text_visualizer_mask_gt, text_visualizer_mask_pred
from NER.utils.metrics import calc_metrics, compute_entity_entropy_per_sample, count_splitting_merging_errors


def check_custom_integration():
    LOAD_MODEL = False
    PLOT = False
    print("Starting custom tests")

    # Load Data
    train, val, test = preprocess_func()
    ul = preprocess_func_ul()

    if LOAD_MODEL:
        H5_MODEL_PATH = "model/ner.h5"
        dir_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(dir_path, H5_MODEL_PATH)

        # loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
        model = tf.keras.models.load_model(model_path)
    else:
        model = TFAutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")

    # for sub in tqdm([train, val, test, ul]):
    for sub in tqdm([test]):
        for i in tqdm(range(1, sub.length), desc="Samples"):
            i = 425
            tokenized_inputs = input_encoder(i, sub)#[None, ...]
            gt = gt_encoder(i, sub)#[None, ...]
            inputs = {}
            inputs["input_ids"] = input_ids(i, sub)[None, ...]
            inputs["token_type_ids"] = input_type_ids(i, sub)[None, ...]
            inputs["attention_mask"] = input_attention_mask(i, sub)[None, ...]

            res = get_sample_topic(i, sub)
            res = metadata_dic(i, sub)

            pred = model(inputs).logits
            pred = tf.transpose(pred, [0, 2, 1])        # simulate as in the platform

            batched_gt = gt[None, ...]
            true_predictions = postprocess_predictions(pred, tokenized_inputs.data["input_ids"])
            true_predictions = postprocess_predictions(pred)
            inputs_ids = inputs["input_ids"][0]
            vis = input_visualizer(inputs_ids)
            visualize(vis) if PLOT else None

            scores = count_splitting_merging_errors(batched_gt, pred)
            scores = calc_metrics(batched_gt, pred)
            res = compute_entity_entropy_per_sample(batched_gt, pred)
            loss = CE_loss(batched_gt, pred)

            vis = text_visualizer_mask_comb(inputs_ids, gt, pred[0])
            visualize(vis) if PLOT else None


            vis = text_visualizer_mask_pred(inputs_ids, pred_vec_labels=pred[0])
            visualize(vis) if PLOT else None


    print("Done")



if __name__ == '__main__':
    check_custom_integration()