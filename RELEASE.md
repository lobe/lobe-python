# Release 0.6.1
___
## Bug Fixes and Other Improvements
Fix super invocations for onnx and tflite backends -- Calling `predict()` was broken because `self.lock` has not been
assigned.


# Release 0.6.0
___
## Breaking Changes
* Refactored the ML backends into sub-folders:
  * `TFModel` class: `backends/backend_tf.py -> backends/tf/backend.py`
  * `TFLiteModel` class: `backends/backend_tflite.py -> backends/tflite/backend.py`
  * `ONNXModel` class: `backends/backend_onnx.py -> backends/onnx/backend.py`

## Bug Fixes and Other Improvements
* Added `Backend` and `ImageBackend` abstract base classes in `backends/backend.py`
* Added ImageBackend classes for each ML backend:
  * `TFImageModel` class: `backends/tf/image_backend.py`
  * `TFLiteImageModel` class: `backends/tflite/image_backend.py`
  * `ONNXImageModel` class: `backends/onnx/image_backend.py`
* Added Grad-CAM++ implementation (`ImageBackend.gradcam_plusplus(image, label) -> np.ndarray`) for visualizing 
convolutional neural network heatmaps for explaining why the model predicted a certain label. 
_Note:_ Grad-CAM++ only implemented currently in `TFImageModel` for TensorFlow Lobe model exports.
The visualization can be called from the top-level API of `ImageModel` -> `ImageModel.visualize(image)`


# Release 0.5.0
___
## Breaking Changes
* Install Lobe with your desired backend options through pip -- `pip install lobe[all]` for everything,
`pip install lobe[tf]` for tensorflow, `pip install --index-url https://google-coral.github.io/py-repo/ tflite_runtime; pip install lobe` for tensorflow lite, `pip install lobe[onnx]` for onnx.


# Release 0.4.0
___
## Bug Fixes and Other Improvements
* Fix API return signature for Lobe 0.9
* Make backwards-compatible with earlier versions of model exports


# Release 0.3.0
___
## Breaking Changes
* Previous use of Signature should be ImageClassificationSignature. `from lobe.signature import Signature` -> 
  `from lobe.signature import ImageClassificationSignature`

## Bug Fixes and Other Improvements
* Update to TensorFlow 2.4 from 1.15.4
* Add ONNX runtime backend
* Use requests instead of urllib
* Make backends thread-safe
* Added constants file for signature keys to enable backwards-compatibility
