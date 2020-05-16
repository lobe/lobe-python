"""
Load a Lobe saved model
"""
from __future__ import annotations
import os
import json
from PIL import Image
from typing import Tuple

from .results import PredictionResult
from ._model import Model
from . import Signature
from . import image_utils

def load_from_signature(signature: Signature) -> ImageModel:
    # Select the appropriate backend
    model_format = signature.format
    if model_format == "tf":
        from .backends import tf_backend as backend
    elif model_format == "tflite":
        from .backends import tflite_backend as backend
    else:
        raise ValueError("Model is an unsupported format")
    
    backend_predict = backend.ImageClassificationModel(signature.model_path)
    return ImageModel(signature, backend_predict)

def load(model_path: str) -> ImageModel:
    # Load the signature
    signature = Signature.load(model_path)

    return load_from_signature(signature)

class ImageModel(Model):
    def __init__(self, signature, backend):
        super(ImageModel, self).__init__(signature)
        self.__backend = backend

    def predict_from_url(self, url: str):
        return self.predict(image_utils.get_image_from_url(url))

    def predict_from_file(self, path: str):
        return self.predict(image_utils.get_image_from_file(path))

    def predict(self, image: Image.Image):
        image_processed = image_utils.preprocess_image(image, self.signature.input_image_size)
        return self.__backend.predict(image_processed)