from __future__ import annotations
from abc import ABC, abstractmethod 
from .Signature import Signature
from ._results import PredictionResult

class Model(ABC):
    def __init__(self, signature: Signature) -> Model:
        self.__signature = signature
        
    @property
    def signature(self) -> Signature:
        return self.__signature

    @abstractmethod
    def predict_from_url(self, url: str):
        pass

    @abstractmethod
    def predict_from_file(self, path: str):
        pass

    @abstractmethod
    def predict(self, input) -> PredictionResult:
        pass