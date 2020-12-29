"""
The constants used for Lobe model exports
"""

# predicted label from the 'outputs' top-level
PREDICTED_LABEL = 'Prediction'
PREDICTED_LABEL_COMPAT = ['Value']  # list of other previous keys to be backwards-compatible
# predicted confidence values from the 'outputs' top-level
LABEL_CONFIDENCES = 'Confidences'
LABEL_CONFIDENCES_COMPAT = ['Labels']  # list of other previous keys to be backwards-compatible

# 'classes' top-level key
CLASSES_KEY = 'classes'

# labels from signature.json 'classes' top-level key
LABELS_LIST = 'Label'

# inputs top-level key
INPUTS = 'inputs'

# outputs top-level return key
OUTPUTS = 'outputs'

ID = 'doc_id'
NAME = 'doc_name'
VERSION = 'doc_version'
FORMAT = 'format'
FILENAME = 'filename'
TAGS = 'tags'

# Input or output properties
TENSOR_NAME = 'name'
TENSOR_SHAPE = 'shape'

IMAGE_INPUT = 'Image'
