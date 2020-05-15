#!/usr/bin/env python
"""
Load a Lobe saved model
"""
from __future__ import annotations
import os
import json
from abc import ABC, abstractmethod
from PIL import Image
from typing import Tuple

from .results import PredictionResult
from . import image_utils

def load(model_path: str) -> ImageModel:
    model_path_real = os.path.realpath(model_path)
    if not os.path.isdir(model_path_real):
        raise ValueError(f"Model folder doesn't exist {model_path}")

    signature_path = os.path.join(model_path_real, "signature.json")
    if not os.path.isfile(signature_path):
        raise ValueError(f"signature.json file not found")

    return ImageModel(model_path)

class ImageModel:
    def __init__(self, model_path: str):
        self.__model_path = model_path

        # Load our signature json file, this shows us the model inputs and outputs.
        # You should open this file and take a look at the inputs/outputs to see their data types, shapes, and names
        with open(os.path.join(model_path, "signature.json"), "r") as f:
            self.__signature = json.load(f)
            
        inputs = self.__signature.get("inputs")
        input_tensor_shape = inputs["Image"]["shape"]

        assert len(input_tensor_shape) == 4
        self.__input_image_size = (input_tensor_shape[1], input_tensor_shape[2])

        model_format = self.__signature.get("format")
        if model_format == "tf":
            from .backends import tf_backend as backend
        elif model_format == "tflite":
            from .backends import tflite_backend as backend
        else:
            raise ValueError("Model is an unsupported format")
        
        self.__backend = backend.ImageClassificationModel(model_path)

    @property
    def model_path(self) -> str:
        return self.__model_path

    @property
    def id(self) -> str:
        return self.__signature.get("doc_id")

    @property
    def name(self) -> str:
        return self.__signature.get("doc_name")

    @property
    def version(self) -> str:
        return self.__signature.get("doc_version")

    @property
    def classes(self):
        classes = []
        if self.__signature.get("classes", None):
            classes = self.__signature.get("classes").get("Label")
        return classes

    @property
    def input_image_size(self) -> Tuple[int, int]:
        return self.__input_image_size

    def __preprocess_image(self, image: Image.Image) -> Image.Image:
        image_processed = image_utils.update_orientation(image)

        # resize and crop image to the model's required size
        image_processed = image_utils.ensure_rgb_format(image)
        image_processed = image_utils.resize_uniform_to_fill(image_processed, self.__input_image_size)
        image_processed = image_utils.crop_center(image_processed, self.__input_image_size)

        return image_processed

    def __str__(self):
        model_info = {
            "id": self.id,
            "name": self.name,
            "version": self.version,
            "classes": self.classes
        }
        return json.dumps(model_info)

    def predict_from_url(self, url: str):
        return self.predict(image_utils.get_image_from_url(url))

    def predict_from_file(self, path: str):
        return self.predict(image_utils.get_image_from_file(path))

    def predict(self, image: Image.Image):
        image_processed = self.__preprocess_image(image)
        return self.__backend.predict(image_processed)
