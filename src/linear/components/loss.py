from abc import ABC, abstractmethod

import numpy as np


class LossFunction(ABC):
    @abstractmethod
    def __call__(self, h: np.ndarray, y: np.ndarray) -> float:
        raise NotImplementedError("Subclasses must implement this method")
    
    @abstractmethod
    def derivative(self, h: np.ndarray, y: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Subclasses must implement this method")
    
    def second_derivative(self, h: np.ndarray, y: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Second derivative not implemented for this loss function")


class MeanSquaredError(LossFunction):
    def __init__(self):
        self.name = 'mean_squared_error'

    def __call__(self, h: np.ndarray, y: np.ndarray) -> float:
        return np.mean(np.power(y - h, 2))

    def derivative(self, h: np.ndarray, y: np.ndarray) -> np.ndarray:
        return 2 * (y - h)

    def second_derivative(self, h: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.full_like(h, 2)


class MeanAbsoluteError(LossFunction):
    def __init__(self):
        self.name = 'mean_absolute_error'

    def __call__(self, h: np.ndarray, y: np.ndarray) -> float:
        return np.mean(np.abs(y - h))

    def derivative(self, h: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.where(y - h < 0, -1, 1)


class Huber(LossFunction):
    def __init__(self, delta: float = 1):
        self.delta = delta
        self.name = 'huber_loss'

    def __call__(self, h: np.ndarray, y: np.ndarray) -> float:
        error = y - h
        quadratic_mask = np.abs(error) <= self.delta
        linear_mask = ~quadratic_mask
        
        quadratic_term = 0.5 * np.power(error[quadratic_mask], 2)
        linear_term = self.delta * np.abs(error[linear_mask]) - 0.5 * (self.delta ** 2)
        
        result = np.zeros_like(error)
        result[quadratic_mask] = quadratic_term
        result[linear_mask] = linear_term
        
        return np.mean(result)

    def derivative(self, h: np.ndarray, y: np.ndarray) -> np.ndarray:
        error = y - h
        return np.where(
            np.abs(error) <= self.delta, 
            error, 
            np.where(error < 0, -self.delta, self.delta)
        )


class LogLoss(LossFunction):
    def __init__(self):
        self.name = 'log_loss'

    def __call__(self, h: np.ndarray, y: np.ndarray) -> float:
        return np.mean(np.log(1 + np.exp(-h * y)))

    def derivative(self, h: np.ndarray, y: np.ndarray) -> np.ndarray:
        return -y / (1 + np.exp(h * y))


class BinaryCrossEntropy(LossFunction):
    def __init__(self, epsilon: float = 1e-9):
        self.name = 'binary_crossentropy'
        self.epsilon = epsilon

    def __call__(self, h: np.ndarray, y: np.ndarray) -> float:
        h_clipped = np.clip(h, self.epsilon, 1 - self.epsilon)
        return -np.mean(y * np.log(h_clipped) + (1 - y) * np.log(1 - h_clipped))

    def derivative(self, h: np.ndarray, y: np.ndarray) -> np.ndarray:
        h_clipped = np.clip(h, self.epsilon, 1 - self.epsilon)
        return (h_clipped - y) / (h_clipped * (1 - h_clipped))


class Hinge(LossFunction):
    def __init__(self, p: int = 1):
        self.name = 'hinge_loss'
        self.p = p

    def __call__(self, h: np.ndarray, y: np.ndarray) -> float:
        margin = 1 - y * h
        return np.mean(np.power(np.maximum(0, margin), self.p))

    def derivative(self, h: np.ndarray, y: np.ndarray) -> np.ndarray:
        margin = 1 - y * h
        mask = margin > 0
        result = np.zeros_like(h)
        
        if self.p == 1:
            result[mask] = -y[mask]
        else:
            result[mask] = -self.p * np.power(margin[mask], self.p - 1) * y[mask]
            
        return -result

    def second_derivative(self, h: np.ndarray, y: np.ndarray) -> np.ndarray:
        if self.p == 2:
            margin = 1 - y * h
            mask = margin > 0
            result = np.zeros_like(h)
            result[mask] = 2 * np.power(y[mask], 2)

            return result

        else:
            raise NotImplementedError("Second derivative only implemented for p=2")