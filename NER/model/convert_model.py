import urllib
import onnx
import tensorflow as tf

from onnx2kerastl import onnx_to_keras


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

