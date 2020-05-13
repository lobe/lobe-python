#!/usr/bin/env python
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

    def __init__(self, labels, prediction: str):
        self.__labels = labels
        self.__prediction = prediction

    @property
    def labels(self):
        return self.__labels

    @property
    def prediction(self) -> str:
        return self.__prediction

    def __str__(self):
        output = {
            "outputs": {
                "Labels": self.labels,
                "Prediction": [self.prediction],
            }
        }
        return json.dumps(output, default=lambda o: o.__dict__)