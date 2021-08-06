from threading import Lock
from ..signature import Signature
from ..signature_constants import TENSOR_NAME
from ..utils import decode_dict_bytes_as_str

try:
    import tflite_runtime.interpreter as tflite

except ImportError:
    # Needs better error text
    raise ImportError(
        "ERROR: This is a TensorFlow Lite model and requires TensorFlow Lite interpreter to be installed on this device. Please install lobe-python with lobe[tflite] or lobe[all] options. If that doesn't work, please go to https://www.tensorflow.org/lite/guide/python and download the appropriate version for you device."
    )


class TFLiteModel(object):
    """
    Generic wrapper for running a TF Lite model exported from Lobe
    """
    def __init__(self, signature: Signature):
        model_path = "{}/{}".format(
            signature.model_path, signature.filename
        )
        self.signature = signature
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        # Combine the information about the inputs and outputs from the signature.json file
        # with the Interpreter runtime details
        input_details = {detail.get("name"): detail for detail in self.interpreter.get_input_details()}
        self.model_inputs = {
            key: {**sig, **input_details.get(sig.get(TENSOR_NAME))}
            for key, sig in self.signature.inputs.items()
        }
        output_details = {detail.get("name"): detail for detail in self.interpreter.get_output_details()}
        self.model_outputs = {
            key: {**sig, **output_details.get(sig.get(TENSOR_NAME))}
            for key, sig in self.signature.outputs.items()
        }
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
            # set the model inputs with our supplied data
            if not isinstance(data, dict):
                # if data isn't a dictionary, set the input to the supplied value
                # throw an error if more than 1 input found and we are only supplied a non-dictionary input
                if len(self.model_inputs) > 1:
                    raise ValueError(
                        f"Found more than 1 model input: {list(self.model_inputs.keys())}, while supplied data wasn't a dictionary: {data}"
                    )
                self.interpreter.set_tensor(list(self.model_inputs.values())[0].get("index"), data)
            else:
                # otherwise, assign data to inputs based on the dictionary
                for input_name, input_detail in self.model_inputs.items():
                    if input_name not in data:
                        raise ValueError(f"Couldn't find input {input_name} in the supplied data {data}")
                    self.interpreter.set_tensor(input_detail.get("index"), data.get(input_name))

            # invoke the interpreter -- runs the model with the set inputs
            self.interpreter.invoke()

            # grab our desired outputs from the interpreter
            # convert to normal python types with tolist()
            outputs = {
                key: self.interpreter.get_tensor(value.get("index")).tolist()
                for key, value in self.model_outputs.items()
            }

            # postprocessing! convert any byte strings to normal strings with .decode()
            decode_dict_bytes_as_str(outputs)
            return outputs
