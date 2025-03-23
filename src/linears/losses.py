import numpy as np


class MeanSquaredError(object):
    def __init__(self):
        self.name = 'mean_squared_error'

    def __call__(self, h: np.ndarray, y: np.ndarray):
        return np.mean(np.power(y - h, 2))

    @staticmethod
    def derivative(h: np.ndarray, y: np.ndarray):
        return 2 * (y - h)

    @staticmethod
    def second_derivative(h: np.ndarray, y: np.ndarray):
        return 2


class MeanAbsoluteError(object):
    def __init__(self):
        self.name = 'mean_absolute_error'

    def __call__(self, h: np.ndarray, y: np.ndarray):
        return np.mean(abs(y - h))

    @staticmethod
    def derivative(h: np.ndarray, y: np.ndarray):
        return -1 if y < h else 1
        # return np.where(y - h < 0, -1, 1)

class Huber(object):
    def __init__(self, delta: float = 1):
        self.delta = delta
        self.name = 'huber_loss'

    def __call__(self, h: np.ndarray, y: np.ndarray):
        error = y - h
        if abs(error) <= self.delta:
            return 0.5 * error ** 2
        elif abs(error) > self.delta:
            return (self.delta * abs(error)) - 0.5 * (self.delta ** 2)

    def derivative(self, h: np.ndarray, y: np.ndarray):
        error = y - h
        if abs(error) <= self.delta:
            return error
        elif abs(error) > self.delta:
            return -self.delta if error < 0 else self.delta


class LogLoss(object):
    def __init__(self):
        self.name = 'log_loss'

    def __call__(self, h: np.ndarray, y: np.ndarray):
        return np.log(1 + np.exp(np.mean(-h * y)))

    @staticmethod
    def derivative(h: np.ndarray, y: np.ndarray):
        return y / (1 + np.exp(h * y))


class BinaryCrossEntropy(object):
    def __init__(self, epsilon: float = 1e-9):
        self.name = 'binary_crossentropy'
        self.epsilon = epsilon

    def __call__(self, h: np.ndarray, y: np.ndarray):
        return -np.mean(y * np.log(h + self.epsilon) + (1 - y) * np.log(1 - h + self.epsilon))

    @staticmethod
    def derivative(h: np.ndarray, y: np.ndarray):
        return y - h


class Hinge(object):
    def __init__(self, p: int = 1):
        self.name = 'hinge_loss'
        self.p = p

    def __call__(self, h: np.ndarray, y: np.ndarray):
        return np.power(max(1 - np.mean(y * h), 0), self.p)

    def derivative(self, h: np.ndarray, y: np.ndarray):
        if np.mean(y * h) >= 1:
            return 0
        else:
            if self.p == 1:
                return y
            else:
                return (self.p * (1 - y * h) ** (self.p - 1)) * y

    def second_derivative(self, h: np.ndarray, y: np.ndarray):
        if self.p == 2:
            return 2 * np.power(y, 2)
        else:
            raise ValueError("Hessian matrix cannot be computed!")