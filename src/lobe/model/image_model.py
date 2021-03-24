"""
Load a Lobe saved model for image classification
"""
from PIL import Image

from .model import Model
from ..signature import ImageClassificationSignature
from .. import image_utils
from ..results import ClassificationResult


class ImageModel(Model):
    signature: ImageClassificationSignature

    @classmethod
    def load_from_signature(cls, signature: ImageClassificationSignature):
        # Select the appropriate backend
        model_format = signature.format
        if model_format == "tf":
            from ..backends.backend_tf import TFModel
            return cls(signature, TFModel(signature))
        elif model_format == "tf_lite":
            from ..backends.backend_tflite import TFLiteModel
            return cls(signature, TFLiteModel(signature))
        elif model_format == "onnx":
            from ..backends.backend_onnx import ONNXModel
            return cls(signature, ONNXModel(signature))
        else:
            raise ValueError(f"Model is an unsupported format: {model_format}")

    @classmethod
    def load(cls, model_path: str):
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
        classification_results = ClassificationResult(
            results=results, labels=self.signature.classes, export_version=self.signature.export_version
        )
        return classification_results
