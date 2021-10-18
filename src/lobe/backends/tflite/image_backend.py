from .backend import TFLiteModel
from ..backend import ImageBackend
from ...signature import ImageClassificationSignature


class TFLiteImageModel(TFLiteModel, ImageBackend):
    def __init__(self, signature: ImageClassificationSignature):
        super(TFLiteModel, self).__init__(signature=signature)

    def gradcam_plusplus(self, image, label=None):
        super(TFLiteModel, self).gradcam_plusplus(image=image, label=label)
