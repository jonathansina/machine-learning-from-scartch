from typing import Tuple
from abc import ABC, abstractmethod

import numpy as np


class LossFunction(ABC):
    @abstractmethod
    @staticmethod
    def compute(error: float, incorrect: np.ndarray, sample_weights: np.ndarray) -> Tuple[float, np.ndarray]:
        raise NotImplementedError("Subclasses must implement this method")
    

class ExponentialLoss(LossFunction):
    @staticmethod
    def compute(error: float, incorrect: np.ndarray, sample_weights: np.ndarray) -> Tuple[float, np.ndarray]:
        estimator_weight = 0.5 * np.log((1 - error) / error)
        new_weights = sample_weights * np.exp(estimator_weight * incorrect)
        return estimator_weight, new_weights
        

class LogisticLoss(LossFunction):
    @staticmethod
    def compute(error: float, incorrect: np.ndarray, sample_weights: np.ndarray) -> Tuple[float, np.ndarray]:
        estimator_weight = np.log((1 - error) / error)
        new_weights = sample_weights / (1 + np.exp(-estimator_weight * incorrect))
        return estimator_weight, new_weights


class SquaredLoss(LossFunction):
    @staticmethod
    def compute(error: float, incorrect: np.ndarray, sample_weights: np.ndarray) -> Tuple[float, np.ndarray]:
        estimator_weight = (1 - error) / error
        new_weights = sample_weights * (1 + estimator_weight * incorrect)
        return estimator_weight, new_weights
