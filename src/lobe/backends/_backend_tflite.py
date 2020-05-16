import numpy as np
from PIL import Image
from .._results import PredictionResult

class ImageClassificationModel():
    __input_key_image = 'Image'
    __input_key_batch_size = "batch_size"
    __output_key_labels = 'Labels_idx_000'
    __output_key_confidences = 'Labels_idx_001'
    __output_key_prediction = 'Prediction'

    def __init__(self, model_path):
        self.__model_path = model_path
        raise ImportError("TFLite not yet supported")

    def __load(self):
        return None

    def predict(self, image: Image.Image) -> PredictionResult:
        return None
