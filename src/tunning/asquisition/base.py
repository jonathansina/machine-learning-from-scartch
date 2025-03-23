from abc import ABC, abstractmethod

import numpy as np
from scipy.stats import norm


class AcquisitionFunction(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass
    
    @abstractmethod
    def __call__(self, y_pred: np.ndarray, y_std: np.ndarray, **kwargs) -> np.ndarray:
        pass

class POI(AcquisitionFunction):
    @property
    def name(self) -> str:
        return "probability_of_improvement"

    def __call__(self, y_pred: np.ndarray, y_std: np.ndarray, **kwargs) -> np.ndarray:
        best_y = kwargs.get('best_y')
        z = (y_pred - best_y) / y_std
        return norm.cdf(z)


class EI(AcquisitionFunction):
    @property
    def name(self) -> str:
        return "expected_improvement"

    def __call__(self, y_pred: np.ndarray, y_std: np.ndarray, **kwargs) -> np.ndarray:
        best_y = kwargs.get('best_y')
        z = (y_pred - best_y) / y_std
        return (y_pred - best_y) * norm.cdf(z) + y_std * norm.pdf(z)


class UCB(AcquisitionFunction):
    def __init__(self, beta: int = 2):
        self._beta = beta

    @property
    def name(self) -> str:
        return "upper_confidence_bound"
    
    @property
    def beta(self) -> int:
        return self._beta

    def __call__(self, y_pred: np.ndarray, y_std: np.ndarray, **kwargs) -> np.ndarray:
        return y_pred + self.beta * y_std
