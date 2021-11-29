from .backend import ONNXModel
from ..backend import ImageBackend
from ...signature import ImageClassificationSignature


class ONNXImageModel(ONNXModel, ImageBackend):
    def __init__(self, signature: ImageClassificationSignature):
        super(ONNXImageModel, self).__init__(signature=signature)

    def gradcam_plusplus(self, image, label=None):
        super(ONNXImageModel, self).gradcam_plusplus(image=image, label=label)
