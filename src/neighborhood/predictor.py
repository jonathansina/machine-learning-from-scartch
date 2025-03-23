from typing import List, Union
from abc import ABC, abstractmethod

import numpy as np


class Predictor(ABC):
    @abstractmethod
    def predict(self, labels: List) -> Union[int, float]:
        raise NotImplementedError("Subclasses must implement this method")


class ClassifierPredictor(Predictor):
    def predict(self, labels: List) -> int:
        return np.argmax(np.bincount(labels))


class RegressorPredictor(Predictor):
    def predict(self, labels: List) -> float:
        return np.mean(labels)