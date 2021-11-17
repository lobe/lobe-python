"""
The constants used for Lobe model exports
"""

# inputs top-level key
INPUTS = 'inputs'

# outputs top-level return key
OUTPUTS = 'outputs'

# predicted label from the 'outputs' top-level (legacy only)
PREDICTED_LABEL_COMPAT = ['Value', 'Prediction']  # list of other previous keys to be backwards-compatible
# predicted confidence values from the 'outputs' top-level
LABEL_CONFIDENCES = 'Confidences'
LABEL_CONFIDENCES_COMPAT = ['Labels']  # list of other previous keys to be backwards-compatible

# 'classes' top-level key
CLASSES_KEY = 'classes'

# labels from signature.json 'classes' top-level key
LABELS_LIST = 'Label'

ID = 'doc_id'
NAME = 'doc_name'
VERSION = 'doc_version'
MODEL_VERSION = 'version'
EXPORT_VERSION = 'export_model_version'
FORMAT = 'format'
FILENAME = 'filename'
TAGS = 'tags'

# Supported model versions
LEGACY_EXPORT_VERSION = -1
SUPPORTED_EXPORT_VERSIONS = [LEGACY_EXPORT_VERSION, 1]

# Input or output properties
TENSOR_NAME = 'name'
TENSOR_SHAPE = 'shape'
TENSOR_DTYPE = 'dtype'

# Special known input key
IMAGE_INPUT = 'Image'

# Values of our model types
TF_MODEL = 'tf'
TFLITE_MODEL = 'tf_lite'
ONNX_MODEL = 'onnx'
