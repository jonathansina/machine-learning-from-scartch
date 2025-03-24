from abc import ABC, abstractmethod

import numpy as np


class ImpurityMeasure(ABC):
    @abstractmethod
    def calculate(self, set_data: np.ndarray) -> float:
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def information_gain(self, data: np.ndarray, left_indices: np.ndarray, right_indices: np.ndarray) -> float:
        raise NotImplementedError("Subclasses must implement this method")