from __future__ import annotations
import json

class PredictionResult():
    @classmethod
    def from_dict(cls, results) -> PredictionResult:
        outputs = results.get('outputs')
        labels = outputs.get('Labels')
        prediction = outputs.get('Prediction')[0]
        return cls(labels, prediction)

    @classmethod
    def from_json(cls, json_str: str) -> PredictionResult:
        results = json.loads(json_str)
        return PredictionResult.from_dict(results)
    
    @classmethod
    def sort_predictions(cls, confidences, labels):
        top_predictions = confidences.argsort()[-5:][::-1]
        labels = labels
        sorted_labels = []
        for i in top_predictions:
            sorted_labels.append(labels[i])
        return sorted_labels, confidences[top_predictions]

    def __init__(self, labels, prediction: str):
        self.__labels = labels
        self.__prediction = prediction

    @property
    def labels(self):
        return self.__labels

    @property
    def prediction(self) -> str:
        return self.__prediction

    def as_dict(self):
        output = {
            "outputs": {
                "Labels": self.labels,
                "Prediction": [self.prediction],
            }
        }
        return output

    def __str__(self) -> str:
        return json.dumps(self.as_dict())