import os
import json
import pathlib
from typing import List, Dict
from .signature_constants import (
    ID, NAME, VERSION, FORMAT, FILENAME, TAGS, CLASSES_KEY, LABELS_LIST, INPUTS, OUTPUTS, IMAGE_INPUT, TENSOR_SHAPE,
    EXPORT_VERSION, LEGACY_EXPORT_VERSION
)


def get_signature_path(model_or_sig_path: str):
    model_or_sig_path = os.path.realpath(os.path.expanduser(model_or_sig_path))

    # This could be a full_path to the signature file
    if os.path.isfile(model_or_sig_path):
        filename, extension = os.path.splitext(model_or_sig_path)
        if (extension.lower() != ".json"):  # Signature file must end in "json"
            raise ValueError(f"Model file provided is not valid: {model_or_sig_path}")
        signature_path = model_or_sig_path  # We have the signature file, so load the model
    elif os.path.isdir(model_or_sig_path):
        # This is a directory with a single Signature File to load
        signature_path = os.path.join(model_or_sig_path, "signature.json")
    else:
        raise ValueError(f"Invalid Signature file or Model directory: {model_or_sig_path}")

    if not os.path.isfile(signature_path):
        raise ValueError(f"signature.json file not found at path: {model_or_sig_path}")

    return pathlib.Path(signature_path)


class Signature(object):
    def __init__(self, model_or_sig_path: str):
        """
        Loads the Signature for the given path to a Lobe signature.json file, or the exported model directory.

        - Use model path when: Using Lobe-Python in its default config, the Signature and TensorFlow (and TFLite) models are expected to be
            in one folder by themselves. Additional models may exist, but they too are expected to be in their own folders.
        - Use signature filepath when: Using Lobe-Python with multiple TensorFlow (and TFLite) models in the same folder, with
            the Signature and Model files named uniquely. This allows you to store multiple TensorFlow/TFLite models and signatures in the same folder.
        """
        # get the signature.json path from the input model or signature path
        signature_path = get_signature_path(model_or_sig_path)
        self.model_path = str(signature_path.parent)

        with open(signature_path, "r", encoding="utf8") as f:
            self._signature = json.load(f)

        self.id: str = self._signature.get(ID)
        self.name: str = self._signature.get(NAME)
        self.version: str = self._signature.get(VERSION)
        self.format: str = self._signature.get(FORMAT)
        self.filename: str = self._signature.get(FILENAME)
        self.tags: List[str] = self._signature.get(TAGS)
        self.export_version: int = self._signature.get(EXPORT_VERSION, LEGACY_EXPORT_VERSION)

        self.inputs: Dict[any, any] = self._signature.get(INPUTS)
        self.outputs: Dict[any, any] = self._signature.get(OUTPUTS)

    def as_dict(self):
        return self._signature

    def __str__(self):
        return json.dumps(self.as_dict())


class ImageClassificationSignature(Signature):
    def __init__(self, model_or_sig_path: str):
        super(ImageClassificationSignature, self).__init__(model_or_sig_path)

        input_tensor_shape: List[int] = self.inputs[IMAGE_INPUT][TENSOR_SHAPE]
        assert len(input_tensor_shape) == 4
        self.input_image_size = (input_tensor_shape[1], input_tensor_shape[2])

        self.classes: List[str] = self._signature.get(CLASSES_KEY, {}).get(LABELS_LIST)
