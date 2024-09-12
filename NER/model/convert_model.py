# from transformers import AutoTokenizer, AutoModelForTokenClassification
# from transformers import pipeline
# from pathlib import Path
# from transformers.onnx import OnnxConfig, export
# from transformers.onnx.features import FeaturesManager
import onnx

import urllib
import onnx
import numpy as np
import tensorflow as tf

from onnx2kerastl import onnx_to_keras
from keras_data_format_converter import convert_channels_first_to_last


if __name__ == "__main__":
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/example-datasets-47ml982d/ner/ner.onnx",
        "ner.onnx")
    onnx_model = onnx.load('ner.onnx')
    input_names = [input.name for input in onnx_model.graph.input]
    keras_model = onnx_to_keras(onnx_model, input_names,
                                input_types=[tf.int32, tf.float32, tf.int32])
    model = keras_model.converted_model

    model.save("ner.h5")
    # model = TFAlbertForQuestionAnswering.from_pretrained("vumichien/albert-base-v2-squad2")
    # question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"
    # inputs = tokenizer(question, text, return_tensors="tf", max_length=384, padding='max_length')

    # Load huggingface model and tokenizer
    # tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER", clean_up_tokenization_spaces=False)
    #
    # model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")

    #
    # nlp = pipeline("ner", model=model, tokenizer=tokenizer)
    # example = "My name is Wolfgang and I live in Berlin"
    #
    # ner_results = nlp(example)
    # print(ner_results)
    #
    #
    # # Convert the model to ONNX format
    # onnx_model_path = Path("bert_base_ner.onnx")
    #
    # # Identify the model's task (token classification)
    # task = "token-classification"
    #
    # # Get the feature extractor
    # onnx_config = OnnxConfig.from_model_config(config=model.config, task='token-classification')
    #
    # # Specify the opset version (usually 11 or 13)
    # opset = 13
    #
    # # Convert the model to ONNX
    # export(
    #     preprocessor=tokenizer,
    #     # tokenizer=tokenizer,
    #     model=model,
    #     output=onnx_model_path,
    #     config=onnx_config,
    #     opset=opset#,
    #     # use_external_format=False
    # )
    #
    # print(f"Model has been converted to ONNX and saved at {onnx_model_path}")

    onnx_model_path = "/Users/daniellebenbashat/TL/leap_hub/NER/NER/model/model.onnx"
    # Load onnx model
    onnx_model = onnx.load(onnx_model_path)


    # outputs = model(**inputs)
    # answer_start_index = int(tf.math.argmax(outputs.start_logits, axis=-1)[0])
    # answer_end_index = int(tf.math.argmax(outputs.end_logits, axis=-1)[0])
    # predict_answer_tokens = inputs.input_ids[0, answer_start_index: answer_end_index + 1]
    # answer = tokenizer.decode(predict_answer_tokens)
    #
    #
    # OnnxConfig.default_fixed_batch = 1
    # OnnxConfig.default_fixed_sequence = 384
    # albert_features = list(FeaturesManager.get_supported_features_for_model_type(model_name_for_features).keys())
    # print(albert_features)
    #
    #
    # onnx_path = Path(onnx_path)
    # if save_model == True:
    #     model_kind, model_onnx_config = FeaturesManager.check_supported_model_or_raise(model,
    #                                                                                    feature='question-answering')
    #     onnx_config = model_onnx_config(model.config)
    #     onnx_inputs, onnx_outputs = export(tokenizer, model, onnx_config, onnx_config.default_onnx_opset, onnx_path)
    # onnx_model = onnx.load(onnx_path)
    # keras_model = onnx_to_keras(onnx_model, ['input_ids', 'token_type_ids', 'attention_mask'],
    #                             input_types=[tf.int32, tf.int32, tf.float32],
    #                             allow_partial_compilation=False)
    # # keras_model = keras_model.converted_model
    # keras_model = tf.keras.models.load_model(
    #     '/Users/daniellebenbashat/TL/leap_hub/squad_albert/model/for_dani_fixed.h5')
    # input_np = [inputs['input_ids'],
    #             inputs['token_type_ids'],
    #             inputs['attention_mask']]
    # out = keras_model(input_np)
    # print(f"len of keras out is {len(out)}")
    # reshaped_outputs = [Reshape((-1, 1))(output) for output in keras_model.outputs]
    # concatenated_output = Concatenate(-1)(reshaped_outputs)
    # # Create a Keras model with the new concatenated output
    # keras_model_ = Model(inputs=keras_model.input, outputs=concatenated_output)
    # out = keras_model_(input_np)
    # print(f"len of keras output after reshaping is {out.shape}")
    # keras_model_.save("newmodel.h5")
    #
    # flipped_model = convert_channels_first_to_last(keras_model_, [])
    # flipped_otpt = flipped_model(input_np)
    # assert np.abs((out[0] - flipped_otpt[1])).max() < 1e-04
    # assert np.abs((out[1] - flipped_otpt[0])).max() < 1e-04



