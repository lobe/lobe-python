"""
Load a Lobe saved model for image classification
"""
from typing import Dict, Union, Optional

import numpy as np
from PIL import Image
from matplotlib import cm
from matplotlib.colors import Colormap

from .model import Model
from .. import image_utils
from ..backends.backend import ImageBackend
from ..signature import ImageClassificationSignature
from ..signature_constants import TF_MODEL, TFLITE_MODEL, ONNX_MODEL
from ..results import ClassificationResult


class VizEnum:
    GRADCAM_PLUSPLUS = 'gradcam_plusplus'
    CNN_FIXATIONS = 'fixations'


class ImageModel(Model):
    signature: ImageClassificationSignature

    @classmethod
    def load_from_signature(cls, signature: ImageClassificationSignature):
        # Select the appropriate backend
        model_format = signature.format
        if model_format == TF_MODEL:
            from ..backends.tf.image_backend import TFImageModel
            return cls(signature, TFImageModel(signature))
        elif model_format == TFLITE_MODEL:
            from ..backends.tflite.image_backend import TFLiteImageModel
            return cls(signature, TFLiteImageModel(signature))
        elif model_format == ONNX_MODEL:
            from ..backends.onnx.image_backend import ONNXImageModel
            return cls(signature, ONNXImageModel(signature))
        else:
            raise ValueError(f"Model is an unsupported format: {model_format}")

    @classmethod
    def load(cls, model_path: str):
        # Load the signature
        return cls.load_from_signature(ImageClassificationSignature(model_path))

    def __init__(self, signature: ImageClassificationSignature, backend: ImageBackend):
        super(ImageModel, self).__init__(signature)
        self.backend = backend

        # register the available visualization functions
        self._viz_functions = {
            VizEnum.GRADCAM_PLUSPLUS: self.backend.gradcam_plusplus,
        }

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

    def visualize(
            self,
            image: Image.Image,
            label: str = None,
            viz: Optional[str] = VizEnum.GRADCAM_PLUSPLUS,
            colormap: Union[str, Colormap] = None,
            opacity=0.5
    ) -> Union[Image.Image, Dict[str, Image.Image]]:
        """
        Visualize what the image classification model is using for its output prediction.
        If label is not supplied, it will use the predicted label from running the model on the image.
        You can optionally supply a matplotlib colormap for colorizing the heatmap on the image.
        You can choose different options for 'viz' to return different methods:
        VizEnum.GRADCAM_PLUSPLUS: Grad-CAM++ (https://arxiv.org/abs/1710.11063) Chattopadhyay et al.
        COMING SOON:
            VizEnum.CNN_FIXATIONS: CNN Fixations (https://arxiv.org/abs/1708.06670) Mopuri et al.
        If 'None' option is given, this will return a dictionary with all options -
        where the key is the same as the options above and the value is the image.
        """
        if viz is not None and viz not in self._viz_functions:
            raise ValueError(
                f"Visualization option `{viz}` not recognized, try one of: {list(self._viz_functions.keys())}."
            )

        image_processed = image_utils.preprocess_image(image, self.signature.input_image_size)
        image_array = image_utils.image_to_array(image_processed)

        viz_return = {}
        for viz_name, viz_func in self._viz_functions.items():
            if viz is None or viz == viz_name:
                viz_heatmap = viz_func(image_array, label)
                viz_img = _image_from_heatmap(
                    heatmap=viz_heatmap,
                    image=image_processed,
                    opacity=opacity,
                    colormap=colormap
                )
                viz_return[viz_name] = viz_img

        if viz is not None:
            viz_return = viz_return.get(viz)

        return viz_return


def _image_from_heatmap(heatmap: np.ndarray, image: Image.Image, opacity=0.5, colormap=None) -> Image.Image:
    """
    Given an activation heatmap (like from Grad-CAM), create a superimposed image of the heatmap
    on the given input image.
    """
    # make sure our heatmap is resized to be same shape as image
    width, height = image.size  # pillow image

    # un-batch the heatmap to a single image
    if heatmap.shape[0] == 1:
        heatmap = np.squeeze(heatmap, axis=0)

    # Use inferno colormap by default to colorize heatmap, unless supplied kwarg
    if colormap is None:
        colormap = "inferno"
    cmap = cm.get_cmap(colormap)

    # Expect heatmap to be floats in 0-1 range, and just grab the rgb values (not rgba)
    color_heatmap = cmap(heatmap)[:, :, :3]

    heatmap_img = image_utils.array_to_image(color_heatmap)

    heatmap_img = heatmap_img.resize((height, width))
    # Return the blended heatmap overlay on the original image
    blended_img = Image.blend(image, heatmap_img, opacity)
    return blended_img
