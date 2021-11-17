import json
from typing import List, Dict

from .api_constants import LABEL, CONFIDENCE, PREDICTIONS
from .signature_constants import (
    PREDICTED_LABEL_COMPAT, LABEL_CONFIDENCES, LABEL_CONFIDENCES_COMPAT, SUPPORTED_EXPORT_VERSIONS
)
from .utils import dict_get_compat

BackendResult = Dict[str, any]


class ClassificationResult:
    """
    Data structure to expose the classification predicted result from running a Lobe model.
    Exposes the top predicted label, as well as a list of tuples (label, confidence) sorted by highest confidence to lowest.

    Sorted list of predictions (label, confidence): ClassificationResult.labels
    Top predicted label: ClassificationResult.prediction

    These can be batched and contain the results for many examples.
    """

    def __init__(self, results: BackendResult, labels: List[str] = None, export_version: int = None):
        """
        Parse the classification results from a dictionary in the form {Name: val} for each output in the signature

        Labels need to be provided to map our confidences, but in the case of the local API they are already returned
        with the prediction.
        """
        # If `results` comes from the Lobe Connect local API, there will not be an export version and the
        # predictions will already be in sorted order. Just need to assign to our 'labels' and 'prediction' variables.
        if export_version is None:
            api_results = results.get(PREDICTIONS, [])
            self.labels = [(prediction.get(LABEL), prediction.get(CONFIDENCE)) for prediction in api_results]
            self.prediction = self.labels[0][0]

        # Otherwise, results comes from running the ImageModel -- check supported versions of the exported model
        elif export_version in SUPPORTED_EXPORT_VERSIONS:
            # grab the list of confidences
            confidences, _ = dict_get_compat(in_dict=results, current_key=LABEL_CONFIDENCES,
                                             compat_keys=LABEL_CONFIDENCES_COMPAT, default=[])
            # zip the labels and confidences together
            labels_and_confidences = []
            # the model results are batched
            for row in confidences:
                # zip this row with the labels to make (label, confidence) pairs
                if not labels:
                    raise ValueError(
                        f"Needed labels to assign the confidences returned. Confidences: {confidences}")
                label_conf_pairs = list(zip(labels, row))
                # sort them by confidence and add to the batch array
                label_conf_pairs = sorted(label_conf_pairs, key=lambda pair: pair[1], reverse=True)
                labels_and_confidences.append(label_conf_pairs)

            # grab the predicted class if it exists (backwards compatibility)
            prediction, _ = dict_get_compat(in_dict=results, current_key=None,
                                            compat_keys=PREDICTED_LABEL_COMPAT)
            # if there was no prediction, grab the label with the highest confidence from labels_and_confidences
            if prediction is None:
                new_prediction = []
                for row in labels_and_confidences:
                    new_prediction.append(row[0][0])
                prediction = new_prediction

            # un-batch if this is a batch size of 1, so that the return is just the value for the single image
            self.labels = _un_batch(labels_and_confidences)
            self.prediction = _un_batch(prediction)

        # Else the exported model version is not officially supported (but may still work anyway)
        # Throw a ValueError with details.
        else:
            raise ValueError(
                f'The model version {export_version} you are using may not be compatible with the supported versions {SUPPORTED_EXPORT_VERSIONS}. Please update both lobe-python and Lobe to latest versions, and try exporting your model again. If the issue persists, please contact us at lobesupport@microsoft.com'
            )

    def as_dict(self):
        return {
            "Labels": self.labels,
            "Prediction": self.prediction,
        }

    def __str__(self) -> str:
        return json.dumps(self.as_dict())


def _un_batch(item):
    """
    Given an arbitrary input, if it is a list with exactly one item then return that first item
    """
    if isinstance(item, list) and len(item) == 1:
        item = item[0]
    return item
