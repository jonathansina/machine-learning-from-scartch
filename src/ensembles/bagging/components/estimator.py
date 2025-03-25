from abc import ABC, abstractmethod

import numpy as np

class Estimator(ABC):
    @abstractmethod
    def fit(self, x: np.ndarray, y: np.ndarray):
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Subclasses must implement this method")