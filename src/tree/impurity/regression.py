import sys
from abc import ABC, abstractmethod

import numpy as np

from path_handler import PathManager

path_manager = PathManager()
sys.path.append(str(path_manager.get_base_directory()))

from src.tree.impurity.base import ImpurityMeasure


class RegressionImpurityMeasure(ImpurityMeasure, ABC):
    @abstractmethod
    def calculate(self, actual: np.ndarray, predicted: float) -> float:
        raise NotImplementedError("Subclasses must implement this method")
    
    @abstractmethod
    def derivative(self, actual: np.ndarray, predicted: float) -> float:
        raise NotImplementedError("Subclasses must implement this method")
    
    def information_gain(self, data: np.ndarray, left_indices: np.ndarray, right_indices: np.ndarray) -> float:
        parent_gain = self.calculate(data[:, -1], np.mean(data[:, -1]))
        
        target_value_left = np.mean(data[left_indices, -1])
        target_value_right = np.mean(data[right_indices, -1])
        
        len_left = len(left_indices)
        len_right = len(right_indices)
        len_total = len(data[:, -1])
        
        left_gain = self.calculate(data[left_indices, -1], target_value_left)
        right_gain = self.calculate(data[right_indices, -1], target_value_right)
        
        child_gain = (len_left / len_total) * left_gain + (len_right / len_total) * right_gain
        return parent_gain - child_gain


class MeanSquaredError(RegressionImpurityMeasure):
    def calculate(self, actual: np.ndarray, predicted: float) -> float:
        return np.mean((actual - predicted) ** 2)
    
    def derivative(self, actual: np.ndarray, predicted: float) -> float:
        return -2 * (actual - predicted)


class MeanAbsoluteError(RegressionImpurityMeasure):
    def calculate(self, actual: np.ndarray, predicted: float) -> float:
        return np.mean(np.abs(actual - predicted))
    
    def derivative(self, actual: np.ndarray, predicted: float) -> float:
        return -np.where(actual - predicted < 0, -1, 1)


class Huber(RegressionImpurityMeasure):
    def __init__(self, delta: float = 1.0):
        self.delta = delta
        
    def calculate(self, actual: np.ndarray, predicted: float) -> float:
        error = actual - predicted
        is_small_error = np.abs(error) <= self.delta
        squared_loss = 0.5 * (error ** 2)
        linear_loss = self.delta * (np.abs(error) - 0.5 * self.delta)

        return np.mean(np.where(is_small_error, squared_loss, linear_loss))
    
    def derivative(self, actual: np.ndarray, predicted: float) -> float:
        error = actual - predicted
        return -np.where(
            np.abs(error) <= self.delta, 
            error, 
            np.where(error < 0, -self.delta, self.delta)
        )