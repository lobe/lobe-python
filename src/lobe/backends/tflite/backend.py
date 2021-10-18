from threading import Lock
from ..backend import Backend
from ...signature import Signature
from ...signature_constants import TENSOR_NAME
from ...utils import decode_dict_bytes_as_str

TFLITE_IMPORT_ERROR = """
ERROR: This is a TensorFlow Lite model and requires TensorFlow Lite interpreter to be installed on this device. 
Please go to https://www.tensorflow.org/lite/guide/python#install_tensorflow_lite_for_python for installation instructions for you device.
"""

# first try to import the tflite interpreter from TensorFlow base library (if we have both installed) to avoid a collision
try:
    import tensorflow.lite as tflite
except ImportError:
    try:
        import tflite_runtime.interpreter as tflite

    except ImportError:
        raise ImportError(TFLITE_IMPORT_ERROR)


class TFLiteModel(Backend):
    """
    Generic wrapper for running a TF Lite model exported from Lobe
    """
    def __init__(self, signature: Signature):
        super(TFLiteModel, self).__init__(signature=signature)
        model_path = "{}/{}".format(
            signature.model_path, signature.filename
        )
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
