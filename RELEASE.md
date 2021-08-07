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
