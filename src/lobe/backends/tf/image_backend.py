from .backend import TFModel, TF_IMPORT_ERROR
from ..backend import ImageBackend
from ...signature import ImageClassificationSignature

import numpy as np

from ...signature_constants import SUPPORTED_EXPORT_VERSIONS, LABEL_CONFIDENCES, LABEL_CONFIDENCES_COMPAT, IMAGE_INPUT, TENSOR_NAME
from ...utils import dict_get_compat

try:
    import tensorflow as tf
except ImportError:
    raise ImportError(TF_IMPORT_ERROR)


class TFImageModel(TFModel, ImageBackend):
    signature: ImageClassificationSignature

    def __init__(self, signature: ImageClassificationSignature):
        super(TFImageModel, self).__init__(signature=signature)
        
    def gradcam_plusplus(self, image: np.ndarray, label=None) -> np.ndarray:
        """
        Implementation of Grad-CAM++,
        adapted for TF 2.x from the original source: https://github.com/adityac94/Grad_CAM_plus_plus

         @article{chattopadhyay2017grad,
          title={Grad-CAM++: Generalized Gradient-based Visual Explanations for Deep Convolutional Networks},
          author={Chattopadhyay, Aditya and Sarkar, Anirban and Howlader, Prantik and Balasubramanian, Vineeth N},
          journal={arXiv preprint arXiv:1710.11063},
          year={2017}
        }
        """
        labels = self.signature.classes
        # Get the output index of the desired label for visualizing
        # If no desired label is given, find the predicted label by running the model on the image
        if label is None:
            label_idx = self._get_predicted_label_argmax(image=image)
        else:
            # if we are batched, get the indices by looping, otherwise just get the index
            if isinstance(label, list):
                label_idx = [labels.index(_label) for _label in label]
            else:
                label_idx = [labels.index(label)]
        # create a one-hot vector of our label indices to use as a mask for the output cost
        label_idx = tf.one_hot(label_idx, depth=len(labels))
        if len(label_idx) != len(image):
            raise ValueError(
                f"Supplied label (or list of labels) does not match the the number of input images. Images : {len(image)}, labels: {len(label_idx)}"
            )

        with self.lock:
            # now we want to get the derivatives of the output with respect to the last conv layer
            # get the layer name of the confidences logits output and the last convolutional layer
            last_fc_tensor, last_conv_tensor = self._get_last_fc_and_conv_tensors()

            # now get the function that returns the fc and conv tensors from the image
            input_image_name = self.signature.inputs[IMAGE_INPUT][TENSOR_NAME]
            last_conv_fn = self.model.prune(input_image_name, last_conv_tensor.name)
            last_fc_fn = self.model.prune(last_conv_tensor.name, last_fc_tensor.name)

            # get the last conv out
            last_conv_out = last_conv_fn(tf.constant(image))

            # take the 3 derivatives of the cost wrt the conv layer
            with tf.GradientTape() as t3:
                t3.watch(last_conv_out)
                with tf.GradientTape() as t2:
                    t2.watch(last_conv_out)
                    with tf.GradientTape() as t1:
                        t1.watch(last_conv_out)
                        last_fc_out = last_fc_fn(last_conv_out)
                        # get the output neuron corresponding to the class of interest
                        cost = last_fc_out * label_idx
                        # first derivative
                        conv_first_grad = t1.gradient(cost, last_conv_out)
                    # second derivative
                    conv_second_grad = t2.gradient(conv_first_grad, last_conv_out)
                # triple derivative
                conv_third_grad = t3.gradient(conv_second_grad, last_conv_out)

            batch, _, _, filters = last_conv_out.shape

            global_sum = tf.math.reduce_sum(last_conv_out, axis=[1, 2])

            alpha_num = conv_second_grad
            broadcasted_global_sum = tf.reshape(global_sum, (batch, 1, 1, filters))
            alpha_denom = conv_second_grad * 2.0 + conv_third_grad * broadcasted_global_sum
            alpha_denom = tf.where(alpha_denom != 0.0, alpha_denom, tf.ones(alpha_denom.shape))
            alphas = alpha_num / alpha_denom

            weights = tf.maximum(conv_first_grad, 0.0)

            alphas_thresholding = tf.where(weights != 0.0, alphas, 0.0)

            alpha_normalization_constant = tf.math.reduce_sum(alphas_thresholding, axis=[1, 2])
            alpha_normalization_constant_processed = tf.where(alpha_normalization_constant != 0.0,
                                                              alpha_normalization_constant,
                                                              tf.ones(alpha_normalization_constant.shape))

            alphas /= tf.reshape(alpha_normalization_constant_processed, (batch, 1, 1, filters))

            deep_linearization_weights = tf.math.reduce_sum((weights * alphas), axis=[1, 2])
            broadcasted_deep_lin_weights = tf.reshape(deep_linearization_weights, (batch, 1, 1, filters))
            grad_CAM_map = tf.math.reduce_sum(broadcasted_deep_lin_weights * last_conv_out, axis=3)

            # Passing through ReLU
            cam = tf.maximum(grad_CAM_map, 0)
            cam_max = tf.math.reduce_max(cam, axis=[1, 2])
            cam /= tf.reshape(cam_max, (batch, 1, 1))  # scale 0 to 1.0
            return cam.numpy()

    def _get_predicted_label_argmax(self, image: np.ndarray):
        """
        Given an image, run our model and return the array of predicted argmax indices.
        """
        result = self.predict(data=image)
        if self.signature.export_version not in SUPPORTED_EXPORT_VERSIONS:
            raise ValueError(
                f"Lobe model export version {self.signature.export_version} not supported. Need one of: {SUPPORTED_EXPORT_VERSIONS}")

        # grab the list of confidences
        confidences, _ = dict_get_compat(
            in_dict=result, current_key=LABEL_CONFIDENCES, compat_keys=LABEL_CONFIDENCES_COMPAT, default=[]
        )
        return tf.argmax(confidences, axis=1)

    def _get_last_fc_and_conv_tensors(self):
        """
        Gets the tensor that represents the last fully-connected layer outputs (logits).
        Since the 'Confidences' output is the softmax tensor, its input is the last FC layer.

        Also find the tensor that is the last convolution layer's output before any pooling (this will be the
        RELU output before the global max or avg pooling)
        """
        confidences_out, _ = dict_get_compat(
            in_dict=self.signature.outputs, current_key=LABEL_CONFIDENCES, compat_keys=LABEL_CONFIDENCES_COMPAT
        )
        softmax_tensor = self.model.graph.get_tensor_by_name(confidences_out.get(TENSOR_NAME))
        # get the op (softmax)'s inputs -- the last fc layer tensor will be the only (first) input to this op
        last_fc_tensor = softmax_tensor.op.inputs[0]

        # now from the last fc layer, bfs search for the closest max pooling op and find its input -- that is the
        # last conv layer output (the RELU tensor)
        last_conv_tensor = None
        visited, queue = [], []
        queue.append(last_fc_tensor)
        while queue:
            tensor = queue.pop()
            visited.append(tensor.name)
            op = tensor.op
            # if this was from the max/avg pool op, get the input tensor
            # (which is the output of the last conv layer's relu)
            if op.type in ["Max", "Mean"]:
                last_conv_tensor = op.inputs[0]
                break
            else:
                for input_tensor in op.inputs:
                    if input_tensor.name not in visited:
                        queue.append(input_tensor)

        return last_fc_tensor, last_conv_tensor
