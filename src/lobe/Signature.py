from __future__ import annotations
import os
import json
import pathlib
from typing import Tuple

def load(model_path: str) -> Signature:
    """
    Loads the Signature for the given PATH or PATH_AND_FILENAME provided:
    - Use PATH When: Using Lobe-Python in its default config, the Signature and TensorFlow (and TFLite) models are expected to be 
        in one folder by themselves. Additional models may exist, but they too are expected to be in their own folders.
    - Use PATH_AND_FILENAME When: Using Lobe-Python with multiple TensorFlow (and TFLite) models in the same folder, with
        the Signature and Model files named uniquely. This allows you to store multiple TensorFlow/TFLite models and signatures in the same folder.
    """
    model_path_real = os.path.realpath(os.path.expanduser(model_path))

    #This could be a full_path to the signature file
    if os.path.isfile(model_path_real):
        filename, extension = os.path.splitext(model_path_real)
        if (extension.lower() != ".json" ): #Signature file must end in "json"
            raise ValueError(f"Model file provided is not valid: {model_path_real}")
        signature_path = model_path_real  #We have the signature file, so load the model
    elif os.path.isdir(model_path_real):
        #This is a directory with a single Signature File to load
        signature_path = os.path.join(model_path_real, "signature.json")
        if not os.path.isfile(signature_path):
            raise ValueError(f"signature.json file not found at path: {model_path}")
    else:
        raise ValueError(f"Invalid Signature file or Model directory: {model_path}")

    return Signature(signature_path) 

class Signature:
    def __init__(self, signature_path: str):
        signature_path = pathlib.Path(signature_path)
        self.__model_path = str(signature_path.parent)

        with open(signature_path, "r") as f:
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