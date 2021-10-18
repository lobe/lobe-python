from threading import Lock

from ..backend import Backend
from ...signature import Signature
from ...utils import decode_dict_bytes_as_str

TF_IMPORT_ERROR = """
ERROR: This is a TensorFlow model and requires tensorflow to be installed on this device. 
Please install lobe-python with lobe[tf] or lobe[all] options. 
If that doesn't work, please go to https://www.tensorflow.org/install for instructions.
"""

try:
    import tensorflow as tf
    from tensorflow.python.training.tracking.tracking import AutoTrackable
except ImportError:
    raise ImportError(TF_IMPORT_ERROR)


class TFModel(Backend):
    """
    Generic wrapper for running a TensorFlow model from Lobe.
    """
    def __init__(self, signature: Signature):
        super(TFModel, self).__init__(signature=signature)
        self.lock = Lock()

        self.model: AutoTrackable = tf.saved_model.load(export_dir=self.signature.model_path, tags=self.signature.tags)
        self.predict_fn = self.model.signatures['serving_default']

    def predict(self, data):
        """
        Predict the outputs by running the data through the model.

        data: can be either a single input value (such as an image array), or a dictionary mapping the input
        keys from the signature to the data they should be assigned

        Returns a dictionary in the form of the signature outputs {Name: value, ...}
        """
        with self.lock:
            # create the feed dictionary that is the input to the model
            feed_dict = {}
            # either map the input data names to the appropriate tensors from the signature inputs, or map to the first
            # input if we are just given a non-dictionary input
            if not isinstance(data, dict):
                # if data isn't a dictionary, set the input to the supplied value
                # throw an error if more than 1 input found and we are only supplied a non-dictionary input
                if len(self.signature.inputs) > 1:
                    raise ValueError(
                        f"Found more than 1 model input: {list(self.signature.inputs.keys())}, while supplied data wasn't a dictionary: {data}"
                    )
                feed_dict[list(self.signature.inputs.keys())[0]] = tf.convert_to_tensor(data)
            else:
                # otherwise, assign data to inputs based on the dictionary
                for input_name in self.signature.inputs.keys():
                    if input_name not in data:
                        raise ValueError(f"Couldn't find input {input_name} in the supplied data {data}")
                    feed_dict[input_name] = tf.convert_to_tensor(data.get(input_name))

            # run the model! there will be as many outputs from session.run as you have in the fetches list
            outputs = self.predict_fn(**feed_dict)

            # postprocessing! make our output dictionary and convert any byte strings to normal strings with .decode()
            results = {}
            for i, (key, tf_val) in enumerate(outputs.items()):
                results[key] = tf_val.numpy().tolist()
            decode_dict_bytes_as_str(results)
            return results
