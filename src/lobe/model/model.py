from abc import ABC, abstractmethod
from ..signature import Signature


class Model(ABC):
    def __init__(self, signature: Signature):
        self.signature = signature

    @abstractmethod
    def predict_from_url(self, url: str):
        pass

    @abstractmethod
    def predict_from_file(self, path: str):
        pass

    @abstractmethod
    def predict(self, input):
        pass
