import os
from leap_binder import preprocess_func, preprocess_func_ul, input_encoder, gt_encoder
import tensorflow as tf

from tqdm import tqdm
from transformers import TFAutoModelForTokenClassification

from NER.ner import *
from leap_binder import *
from code_loader.helpers import visualize


def check_custom_integration():
    LOAD_MODEL = True
    PLOT = True
    print("Starting custom tests")

    # Load Data
    train, val, test = preprocess_func()
    ul = preprocess_func_ul()

    if LOAD_MODEL:
        H5_MODEL_PATH = "NER/NER/model/ner.h5"
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

            res = metadata_dic(i, train)

            # pred = model(inputs).logits
            pred = model(inputs)
            pred = tf.transpose(pred, [0, 2, 1]).numpy()  # simulate as in the platform

            batched_gt = gt[None, ...]
            line1, line2 = hf_decode_labels(train.data['ds'][0])
            true_predictions = postprocess_predictions(pred, tokenized_inputs.data["input_ids"])
            true_predictions = postprocess_predictions(pred)

            scores = count_splitting_merging_errors(batched_gt, pred)
            scores = calc_metrics(batched_gt, pred)
            res = compute_entity_entropy_per_sample(batched_gt, pred)
            loss = CE_loss(batched_gt, pred)

            # vis

            inputs_ids = inputs["input_ids"]
            vis = input_visualizer(inputs_ids)
            visualize(vis) if PLOT else None

            vis = loss_visualizer(inputs_ids, batched_gt, pred)
            visualize(vis) if PLOT else None

            vis = text_visualizer_mask_pred(inputs_ids, pred_vec_labels=pred)
            visualize(vis) if PLOT else None

            vis = text_visualizer_mask_gt(inputs_ids, batched_gt)
            visualize(vis) if PLOT else None

            vis = text_visualizer_mask_comb(inputs_ids, batched_gt, pred)
            visualize(vis) if PLOT else None

    print("Done")



if __name__ == '__main__':
    check_custom_integration()