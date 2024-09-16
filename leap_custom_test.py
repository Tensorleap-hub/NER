import os
from leap_binder import preprocess_func, preprocess_func_ul, input_encoder, gt_encoder
import tensorflow as tf

from tqdm import tqdm
from transformers import TFAutoModelForTokenClassification

from NER.ner import *
from leap_binder import *
# from code_loader.helpers import visualize


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

    for sub in tqdm([train, val, test, ul]):
        for i in tqdm(range(1, sub.length), desc="Samples"):
            tokenized_inputs = input_encoder(i, train)#[None, ...]
            gt = gt_encoder(i, train)#[None, ...]
            inputs = {}
            inputs["input_ids"] = input_ids(i, train)[None, ...]
            inputs["token_type_ids"] = input_type_ids(i, train)[None, ...]
            inputs["attention_mask"] = input_attention_mask(i, train)[None, ...]

            res = metadata_dic(i, train)

            pred = model(inputs).logits
            pred = tf.transpose(pred, [0, 2, 1])        # simulate as in the platform

            batched_gt = gt[None, ...]
            line1, line2 = hf_decode_labels(train.data['ds'][0])
            true_predictions = postprocess_predictions(pred, tokenized_inputs.data["input_ids"])
            true_predictions = postprocess_predictions(pred)
            # postprocess_labels(gt[None, ...].numpy().tolist())
            inputs_ids = inputs["input_ids"][0]
            vis = input_visualizer(inputs_ids)
            vis.plot_visualizer() if PLOT else None
            # visualize(vis) if PLOT else None

            vis = text_visualizer_mask_pred(inputs_ids, pred_vec_labels=pred[0])
            vis.plot_visualizer() if PLOT else None
            #
            vis = text_visualizer_mask_gt(inputs_ids, gt_vec_labels=gt)
            vis.plot_visualizer() if PLOT else None
            #
            # vis = class_visualizer(pred)
            # vis.plot_visualizer() if PLOT else None
            #
            # vis = class_visualizer(gt)
            # vis.plot_visualizer() if PLOT else None
            loss = CE_loss(batched_gt, pred)
            scores = precision_recall_f1(batched_gt, pred)

    print("Done")


        # choices = choice_encoder(i, x[1])[None, ...]
    #     gt = gt_encoder(i, x[1])[None, ...]
    #     print(img.shape)
    #     print(question.shape)
    #     print(choices.shape)
    #     vis_image = image_visualizer(img[0])
    #     # plt.imshow(vis_image.data)
    #     decoded_text = question_visualizer(question[0])
    #     decoded_choice = choice_visualizer(choices[0])
    #     choice_gt_vis_ = choice_gt_vis(choices[0], gt)
    #     res = model([question, img, choices[..., 0]])
    #     metadata_dict = get_metadata(i, x[1])
    #     # ls = loss(tf.nn.softmax(res), gt)
    #     metadata_q = question_metadata(i, x[1])
    #     metadata_skills = skills_metadata(i, x[1])


if __name__ == '__main__':
    check_custom_integration()