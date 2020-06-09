import numpy as np
from PIL import Image
from .._results import PredictionResult

try:
    import tflite_runtime.interpreter as tflite

except ImportError:
    # Needs better error text
    raise ImportError(
        "ERROR: This is a TensorFlow Lite model and requires TensorFlow Lite interpreter to be installed on this device. Please go to https://www.tensorflow.org/lite/guide/python and download the appropriate version for you device."
    )


class ImageClassificationModel:
    __MAX_UINT8 = 255

    def __init__(self, signature):
        self.__model_path = "{}/{}".format(
            signature.model_path, signature.filename
        )
        self.__tflite_predict_fn = None
        self.__labels = signature.classes

        raise ImportError("TFLite not yet supported")

    def __load(self):
        self.__tflite_predict_fn = tflite.Interpreter(
            model_path=self.__model_path
        )

    def predict(self, image: Image.Image) -> PredictionResult:
        if self.__tflite_predict_fn is None:
            self.__load()

        self.__tflite_predict_fn.allocate_tensors()

        # Add an extra axis onto the numpy array
        np_image = np.expand_dims(image, axis=0)

        # Converts to floating point and standardize range from 0 to 1.
        np_image = np.float32(np_image) / self.__MAX_UINT8

        input_details = self.__tflite_predict_fn.get_input_details()
        output_details = self.__tflite_predict_fn.get_output_details()

        self.__tflite_predict_fn.set_tensor(
            input_details[0]["index"],
            np_image
        )

        self.__tflite_predict_fn.invoke()
        top_prediction_output = self.__tflite_predict_fn.get_tensor(
            output_details[0]["index"]
        )

        confidences_output = self.__tflite_predict_fn.get_tensor(
            output_details[1]["index"]
        )

        confidences = np.squeeze(confidences_output)
        top_prediction = top_prediction_output.item().decode("utf-8")

        return PredictionResult(
            labels=self.__labels,
            confidences=confidences,
            prediction=top_prediction,
        )
