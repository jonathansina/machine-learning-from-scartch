import math
from abc import ABC, abstractmethod

import numpy as np


class DistanceMetric(ABC):
    def __init__(self):
        self.name = "distance"
    
    @abstractmethod
    def __call__(self, x: np.ndarray, y: np.ndarray) -> float:
        raise NotImplementedError("Method should be implemented in the child class")


class Euclidean(DistanceMetric):
    def __init__(self):
        self.name = "euclidean_distance"

    def __call__(self, x: np.ndarray, y: np.ndarray) -> float:
        dist = [(a - b) ** 2 for a, b in zip(x, y)]
        dist = math.sqrt(sum(dist))
        return dist


class Manhattan(DistanceMetric):
    def __init__(self):
        self.name = "manhattan_distance"

    def __call__(self, x: np.ndarray, y: np.ndarray) -> float:
        dist = [abs(a - b) for a, b in zip(x, y)]
        return sum(dist)


class Chebyshev(DistanceMetric):
    def __init__(self):
        self.name = "chebyshev_distance"

    def __call__(self, x: np.ndarray, y: np.ndarray) -> float:
        dist = [abs(a - b) for a, b in zip(x, y)]
        return max(dist)


class Minkowski(DistanceMetric):
    def __init__(self, p: int = 2):
        self.name = "minkowski_distance"
        self.p = p

    def __call__(self, x, y) -> float:
        return sum([(a - b) ** self.p for a, b in zip(x, y)]) ** 1 / self.p


class Hamming(DistanceMetric):
    def __init__(self):
        self.name = "hamming_distance"

    def __call__(self, x: np.ndarray, y: np.ndarray) -> float:
        if len(x) != len(y):
            raise ValueError("Two vectors must have the same length!")
        distance = 0
        for a, b in zip(x, y):
            if a != b:
                distance += 1
        return distance / len(x)


class Cosine(DistanceMetric):
    def __init__(self):
        self.name = "cosine_distance"

    def __call__(self, x: np.ndarray, y: np.ndarray) -> float:
        numerator = np.dot(x, y)
        denominator = np.linalg.norm(x) * np.linalg.norm(y)
        return 1 - numerator / denominator


class Jaccard(DistanceMetric):
    def __init__(self):
        self.name = "jaccard_distance"

    def __call__(self, x: np.ndarray, y: np.ndarray) -> float:
        intersection = len(set(x).intersection(set(y)))
        union = len(set(x).union(set(y)))
        return 1 - intersection / union
