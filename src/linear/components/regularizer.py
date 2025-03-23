from abc import ABC, abstractmethod

import numpy as np


class Regularizer(ABC):
    def __init__(self, lambda_: float = 1.0):
        self.lambda_ = lambda_
        
    @abstractmethod
    def __call__(self, weights: np.ndarray) -> float:
        raise NotImplementedError("Subclasses must implement this method")
    
    @abstractmethod
    def derivative(self, weights: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Subclasses must implement this method")


class L1Regularizer(Regularizer):
    def __call__(self, weights: np.ndarray) -> float:
        return self.lambda_ * np.sum(np.abs(weights))
    
    def derivative(self, weights: np.ndarray) -> np.ndarray:
        return self.lambda_ * np.sign(weights)


class L2Regularizer(Regularizer):
    def __call__(self, weights: np.ndarray) -> float:
        return self.lambda_ * np.sum(np.square(weights))
    
    def derivative(self, weights: np.ndarray) -> np.ndarray:
        return self.lambda_ * 2 * weights


class ElasticNetRegularizer(Regularizer):
    def __init__(self, lambda_: float = 1.0, alpha: float = 0.8):
        super().__init__(lambda_)
        self.alpha = alpha
    
    def __call__(self, weights: np.ndarray) -> float:
        l1_term = self.alpha * np.sum(np.abs(weights))
        l2_term = (1 - self.alpha) * np.sum(np.square(weights))
        return self.lambda_ * (l1_term + l2_term)
    
    def derivative(self, weights: np.ndarray) -> np.ndarray:
        l1_term = self.alpha * np.sign(weights)
        l2_term = 2 * (1 - self.alpha) * weights
        return self.lambda_ * (l1_term + l2_term)