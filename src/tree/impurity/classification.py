import sys
from abc import ABC, abstractmethod

import numpy as np

from path_handler import PathManager

path_manager = PathManager()
sys.path.append(str(path_manager.get_base_directory()))

from src.tree.impurity.base import ImpurityMeasure


class ClassificationImpurityMeasure(ImpurityMeasure, ABC):
    @abstractmethod
    def calculate(self, set_data: np.ndarray) -> float:
        raise NotImplementedError("Subclasses must implement this method")
    
    def information_gain(self, data: np.ndarray, left_indices: np.ndarray, right_indices: np.ndarray) -> float:
        parent_gain = self.calculate(data[:, -1])
        
        len_left = len(left_indices)
        len_right = len(right_indices)
        len_total = len(data[:, -1])
        
        left_gain = self.calculate(data[left_indices, -1])
        right_gain = self.calculate(data[right_indices, -1])
        
        child_gain = (len_left / len_total) * left_gain + (len_right / len_total) * right_gain
        return parent_gain - child_gain


class Gini(ClassificationImpurityMeasure):
    def calculate(self, set_data: np.ndarray) -> float:
        set_data = np.array(set_data, dtype=np.int64)
        prob = np.bincount(set_data) / len(set_data)
        return sum(prob * (1 - prob))


class Entropy(ClassificationImpurityMeasure):
    def calculate(self, set_data: np.ndarray) -> float:
        set_data = np.array(set_data, dtype=np.int64)
        prob = np.bincount(set_data) / len(set_data)
        return -sum(p * np.log2(p) for p in prob if p > 0)