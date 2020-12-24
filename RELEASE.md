# Release 0.3.0
___
## Breaking Changes
* Previous use of Signature should be ImageClassificationSignature. `from lobe.signature import Signature` -> 
  `from lobe.signature import ImageClassificationSignature`

## Bug Fixes and Other Improvements
* Update to TensorFlow 2.4 from 1.15.4
* Use requests instead of urllib
* Make backends thread-safe
* Added constants file for signature keys to enable backwards-compatibility