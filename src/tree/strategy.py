from typing import Any
from abc import ABC, abstractmethod

import numpy as np


class TreeBuilderStrategy(ABC):
    @abstractmethod
    def determine_leaf_value(self, data: np.ndarray) -> Any:
        raise NotImplementedError("Subclasses must implement this method")


class ClassificationTreeBuilder(TreeBuilderStrategy):
    @staticmethod
    def determine_leaf_value(data: np.ndarray) -> int:
        y = np.array(data[:, -1], dtype=np.int64)
        return np.argmax(np.bincount(y))


class RegressionTreeBuilder(TreeBuilderStrategy):
    @staticmethod
    def determine_leaf_value(data: np.ndarray) -> float:
        return np.mean(data[:, -1])