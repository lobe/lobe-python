from __future__ import annotations
import json
from typing import List, Dict

from .signature_constants import (
    PREDICTED_LABEL, PREDICTED_LABEL_COMPAT, LABEL_CONFIDENCES, LABEL_CONFIDENCES_COMPAT, OUTPUTS
)
from .utils import dict_get_compat, list_or_tuple


class ClassificationResult:
    """
    Data structure to expose the classification predicted result from running a Lobe model.
    Exposes the top predicted label, as well as a list of tuples (label, confidence) sorted by highest confidence to lowest.

    Sorted list of predictions (label, confidence): ClassificationResult.labels
    Top predicted label: ClassificationResult.prediction

    These can be batched and contain the results for many examples.
    """

    @classmethod
    def from_json(cls, json_str: str, labels: List[str] = None) -> ClassificationResult:
        """
        Parse the classification results from a json string
        """
        results = json.loads(json_str)
        return cls(results=results, labels=labels)

    def __init__(self, results: Dict[str, any], labels: List[str] = None):
        """
        Parse the classification results from a dictionary in the form {Name: val} for each output in the signature

        Labels need to be provided to map our confidences, but in the case of the local API they are already returned
        with the prediction.
        """
        # first check if this results dictionary started with the 'outputs' top-level key and navigate inside if true
        outputs_dict = results.get(OUTPUTS)
        if outputs_dict is not None:
            results = outputs_dict

        # grab the list of confidences -- this may or may not include labels
        confidences, _ = dict_get_compat(in_dict=results, current_key=LABEL_CONFIDENCES,
                                         compat_keys=LABEL_CONFIDENCES_COMPAT, default=[])
        # zip the labels and confidences together. labels can be None (they should already exist in confidences)
        labels_and_confidences = []
        # if confidences were unbatched and already contain (label, confidence) pairs (as when returned from local API)
        # just sort the confidences
        if _is_label_conf_pair(confidences):
            labels_and_confidences = sorted(confidences, key=lambda pair: pair[1], reverse=True)
        else:
            # otherwise, it could be batched -- go deeper!
            for row in confidences:
                # if this row is a list of numbers, zip it with the labels to make (label, confidence) pairs
                if list_or_tuple(row) and len(row) > 0 and isinstance(row[0], float):
                    if not labels:
                        raise ValueError(f"Needed labels to assign the confidences returned. Confidences: {confidences}")
                    label_conf_pairs = list(zip(labels, row))
                    # sort them by confidence
                    label_conf_pairs = sorted(label_conf_pairs, key=lambda pair: pair[1], reverse=True)
                    labels_and_confidences.append(label_conf_pairs)
                else:
                    # if this row is (label, confidence) pairs already, sort them
                    if list_or_tuple(row) and len(row) > 0 and list_or_tuple(row[0]) and len(row[0]) == 2 and isinstance(row[0][0], str) and isinstance(row[0][1], float):
                        label_conf_pairs = sorted(row, key=lambda pair: pair[1], reverse=True)
                        labels_and_confidences.append(label_conf_pairs)
                    else:
                        raise ValueError(f"Found unexpected confidence return: {confidences}")

        # grab the predicted class if it exists
        prediction, _ = dict_get_compat(in_dict=results, current_key=PREDICTED_LABEL,
                                        compat_keys=PREDICTED_LABEL_COMPAT)
        # if there was no prediction, grab the label with the highest confidence from labels_and_confidences
        if prediction is None:
            new_prediction = []
            for row in labels_and_confidences:
                new_prediction.append(row[0][0])
            prediction = new_prediction

        # un-batch if this is a batch size of 1, so that the return is just the value for the single image
        labels_and_confidences = _un_batch(labels_and_confidences)

        # un-batch if this is a batch size of 1, so that the return is just the value for the single image
        prediction = _un_batch(prediction)

        self.labels = labels_and_confidences
        self.prediction = prediction

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


def _is_label_conf_pair(row):
    return (
            list_or_tuple(row) and len(row) > 0 and
            list_or_tuple(row[0]) and len(row[0]) == 2 and
            isinstance(row[0][0], str) and isinstance(row[0][1], float)
    )
