from threading import Lock

from ..signature import Signature
from ..signature_constants import TENSOR_NAME
from ..utils import decode_dict_bytes_as_str

try:
    import tensorflow as tf
except ImportError:
    raise ImportError("ERROR: This is a TensorFlow model and requires tensorflow to be installed on this device. Please run\n\tpip install tensorflow==1.15.4\n")


class TFModel(object):
    """
    Generic wrapper for running a tensorflow model
    """
    def __init__(self, signature: Signature):
        self.lock = Lock()
        self.signature = signature

        # placeholder for the tensorflow session
        self.session = None
        # load the model initially
        self.load()

    def load(self):
        self.cleanup()
        with self.lock:
            # create a new tensorflow session
            self.session = tf.compat.v1.Session(graph=tf.Graph())
            # load our model into the session
            tf.compat.v1.saved_model.loader.load(sess=self.session, tags=self.signature.tags,
                                                     export_dir=self.signature.model_path)

    def predict(self, data):
        """
        Predict the outputs by running the data through the model.

        data: can be either a single input value (such as an image array), or a dictionary mapping the input
        keys from the signature to the data they should be assigned

        Returns a dictionary in the form of the signature outputs {Name: value, ...}
        """
        # load the model if we don't have a session
        if self.session is None:
            self.load()

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
                feed_dict[list(self.signature.inputs.values())[0].get(TENSOR_NAME)] = data
            else:
                # otherwise, assign data to inputs based on the dictionary
                for input_name, input_sig in self.signature.inputs.items():
                    if input_name not in data:
                        raise ValueError(f"Couldn't find input {input_name} in the supplied data {data}")
                    feed_dict[input_sig.get(TENSOR_NAME)] = data.get(input_name)

            # list the outputs we want from the model -- these come from our signature
            # since we are using dictionaries that could have different orders, make tuples of (key, name) to keep track for putting
            # the results back together in a dictionary
            fetches = [(key, output.get(TENSOR_NAME)) for key, output in self.signature.outputs.items()]

            # run the model! there will be as many outputs from session.run as you have in the fetches list
            outputs = self.session.run(fetches=[name for _, name in fetches], feed_dict=feed_dict)

            # postprocessing! make our output dictionary and convert any byte strings to normal strings with .decode()
            results = {}
            for i, (key, _) in enumerate(fetches):
                results[key] = outputs[i].tolist()
            decode_dict_bytes_as_str(results)
            return results

    def cleanup(self):
        """
        When you are done, free up memory by closing the TF session in this cleanup function
        """
        with self.lock:
            # close our tensorflow session if one exists
            if self.session is not None:
                self.session.close()
                self.session = None

    def __del__(self):
        self.cleanup()
