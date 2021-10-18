from threading import Lock
from ...signature import Signature
from ...signature_constants import TENSOR_NAME
from ...utils import decode_dict_bytes_as_str

ONNX_IMPORT_ERROR = """
ERROR: This is an ONNX model and requires onnx runtime to be installed on this device. 
Please install lobe-python with lobe[onnx] or lobe[all] options. 
If that doesn't work, please go to https://www.onnxruntime.ai/ for install instructions.
"""

try:
    import onnxruntime as rt

except ImportError:
    # Needs better error text
    raise ImportError(ONNX_IMPORT_ERROR)


class ONNXModel(object):
    """
    Generic wrapper for running an ONNX model exported from Lobe
    """
    def __init__(self, signature: Signature):
        model_path = "{}/{}".format(
            signature.model_path, signature.filename
        )
        self.signature = signature

        # load our onnx inference session
        self.session = rt.InferenceSession(path_or_bytes=model_path)

        self.lock = Lock()

    def predict(self, data):
        """
        Predict the outputs by running the data through the model.

        data: can be either a single input value (such as an image array), or a dictionary mapping the input
        keys from the signature to the data they should be assigned

        Returns a dictionary in the form of the signature outputs {Name: value, ...}
        """
        # make the predict function thread-safe
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

            # run the model!
            # get the outputs
            fetches = [(key, value.get("name")) for key, value in self.signature.outputs.items()]
            outputs = self.session.run(output_names=[name for (_, name) in fetches], input_feed=feed_dict)
            # make our return a dict from the list of outputs that correspond to the fetches
            results = {}
            for i, (key, _) in enumerate(fetches):
                results[key] = outputs[i].tolist()
            # postprocessing! convert any byte strings to normal strings with .decode()
            decode_dict_bytes_as_str(results)
            return results
