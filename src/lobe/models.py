#!/usr/bin/env python
"""
Load a Lobe TF saved model
"""
from __future__ import annotations
import os
import json
from abc import ABC, abstractmethod
import tensorflow as tf
from tensorflow.contrib import predictor
from PIL import Image
import numpy as np
from typing import Tuple

from results import PredictionResult
import image_utils

class ImageModel(ABC):
    @classmethod
    def from_path(cls, model_path: str) -> ImageModel:
        model_path_real = os.path.realpath(model_path)
        if not os.path.isdir(model_path_real):
            raise ValueError(f"Model folder doesn't exist {model_path}")

        signature_path = os.path.join(model_path_real, "signature.json")
        if not os.path.isfile(signature_path):
            raise ValueError(f"signature.json file not found")

        return ImageClassificationModel(model_path_real)

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

    def preprocess_image(self, image: Image.Image) -> np.ndarray:
        image_processed = image_utils.update_orientation(image)

        # resize and crop image to the model's required size
        image_processed = image_utils.ensure_rgb_format(image)
        image_processed = image_utils.resize_uniform_to_fill(image_processed, self.__input_image_size)
        image_processed = image_utils.crop_center(image_processed, self.__input_image_size)

        # Convert all values to range [0,1]
        np_image = np.asarray(image_processed) / 255.

        # Finally, add an extra axis onto the numpy array
        return np_image[np.newaxis, ...]

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

    @abstractmethod
    def predict(self, image: Image.Image):
        pass

class ImageClassificationModel(ImageModel):
    __input_key_image = 'Image'
    __input_key_batch_size = "batch_size"
    __output_key_labels = 'Labels_idx_000'
    __output_key_confidences = 'Labels_idx_001'
    __output_key_prediction = 'Prediction'

    def __init__(self, model_path):
        super().__init__(model_path)

        # placeholder for the tensorflow predictor
        self.predict_fn = None

    def __load(self):
        self.predict_fn = predictor.from_saved_model(self.model_path)

    def predict(self, image: Image.Image):
        if self.predict_fn is None:
            self.__load()

        np_image = self.preprocess_image(image)

        predictions = self.predict_fn({
                self.__input_key_image: np_image,
                self.__input_key_batch_size: 1 })

        labels = [label.decode('utf-8') for label in predictions[self.__output_key_labels][0].tolist()]
        confidences = predictions[self.__output_key_confidences][0].tolist()
        top_prediction = predictions[self.__output_key_prediction][0].decode('utf-8')
        
        return PredictionResult(labels=list(zip(labels, confidences)), prediction=top_prediction)
