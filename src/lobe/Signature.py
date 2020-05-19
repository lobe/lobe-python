from __future__ import annotations
import os
import json
from typing import Tuple

def load(model_path: str) -> Signature:
    model_path_real = os.path.realpath(os.path.expanduser(model_path))
    if not os.path.isdir(model_path_real):
        raise ValueError(f"Model directory does not exist: {model_path}")

    signature_path = os.path.join(model_path_real, "signature.json")
    if not os.path.isfile(signature_path):
        raise ValueError(f"signature.json file not found at path: {model_path}")

    return Signature(model_path_real)

class Signature:
    def __init__(self, model_path: str):
        self.__model_path = model_path

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
    def format(self) -> str:
        return self.__signature.get("format")

    @property
    def filename(self) -> str:
        return self.__signature.get("filename")

    @property
    def input_tensor_shape(self) -> str:
        return self.__input_image_size

    @property
    def classes(self):
        classes = []
        if self.__signature.get("classes", None):
            classes = self.__signature.get("classes").get("Label")
        return classes

    @property
    def input_image_size(self) -> Tuple[int, int]:
        return self.__input_image_size

    def as_dict(self):
        return self.__signature

    def __str__(self):
        return json.dumps(self.as_dict())