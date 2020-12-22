"""
Load a Lobe saved model for image classification
"""
from __future__ import annotations
from PIL import Image

from .model import Model
from ..signature import ImageClassificationSignature
from .. import image_utils
from ..results import ClassificationResult


class ImageModel(Model):
    signature: ImageClassificationSignature

    @classmethod
    def load_from_signature(cls, signature: ImageClassificationSignature) -> ImageModel:
        # Select the appropriate backend
        model_format = signature.format
        if model_format == "tf":
            from ..backends.backend_tf import TFModel
            return cls(signature, TFModel(signature))
        elif model_format == "tf_lite":
            from ..backends.backend_tflite import TFLiteModel
            return cls(signature, TFLiteModel(signature))
        else:
            raise ValueError("Model is an unsupported format")

    @classmethod
    def load(cls, model_path: str) -> ImageModel:
        # Load the signature
        return cls.load_from_signature(ImageClassificationSignature(model_path))

    def __init__(self, signature: ImageClassificationSignature, backend):
        super(ImageModel, self).__init__(signature)
        self.backend = backend

    def predict_from_url(self, url: str):
        return self.predict(image_utils.get_image_from_url(url))

    def predict_from_file(self, path: str):
        return self.predict(image_utils.get_image_from_file(path))

    def predict(self, image: Image.Image) -> ClassificationResult:
        image_processed = image_utils.preprocess_image(image, self.signature.input_image_size)
        image_array = image_utils.image_to_array(image_processed)
        results = self.backend.predict(image_array)
        classification_results = ClassificationResult(results=results, labels=self.signature.classes)
        return classification_results
