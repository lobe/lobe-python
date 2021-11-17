"""
Abstract for our backend implementations.
"""
from abc import ABC, abstractmethod

from ..signature import Signature
from ..results import BackendResult


class Backend(ABC):
	def __init__(self, signature: Signature):
		self.signature = signature

	@abstractmethod
	def predict(self, data: any) -> BackendResult:
		"""
		Predict the outputs by running the data through the model.

		data: can be either a single input value (such as an image array), or a dictionary mapping the input
		keys from the signature to the data they should be assigned

		Returns a dictionary in the form of the signature outputs {Name: value, ...}
		"""
		pass


class ImageBackend(Backend):
	def gradcam_plusplus(self, image, label: str = None):
		"""
		Return the heatmap from Grad-CAM++
		https://arxiv.org/abs/1710.11063
		Grad-CAM++: Improved Visual Explanations for Deep Convolutional Networks
		Aditya Chattopadhyay, Anirban Sarkar, Prantik Howlader, Vineeth N Balasubramanian
		"""
		raise NotImplementedError(
			f"Image backend {self.__class__.__name__} doesn't have a Grad-CAM++ implementation yet."
		)
