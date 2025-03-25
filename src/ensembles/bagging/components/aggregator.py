from typing import Tuple
from abc import ABC, abstractmethod

import numpy as np


class PredictionAggregator(ABC):
    @abstractmethod
    def aggregate(self, predictions: np.ndarray) -> Tuple[float, float]:
        raise NotImplementedError("Subclasses must implement this method")


class ClassifierAggregator(PredictionAggregator):
    def aggregate(self, predictions: np.ndarray) -> Tuple[float, float]:
        y = np.array(predictions, dtype=np.int64)
        result = np.argmax(np.bincount(y))
        accuracy = np.bincount(y)[result] / len(y) * 100

        return result, accuracy


class RegressorAggregator(PredictionAggregator):
    def aggregate(self, predictions: np.ndarray) -> Tuple[float, float]:
        result = np.mean(predictions)
        accuracy = np.std(predictions)

        return result, accuracy
