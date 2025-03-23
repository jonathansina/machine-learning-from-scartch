import sys
from typing import Union

import numpy as np

from path_handler import PathManager

path_manager = PathManager()
sys.path.append(str(path_manager.get_base_directory()))

from src.neighborhoods.distances import (
    Cosine, 
    Hamming,
    Jaccard, 
    Manhattan, 
    Chebyshev,
    Euclidean, 
    Minkowski,
    DistanceMetric
)
from src.neighborhoods.search import BruteForceSearch, KDTreeSearch
from src.neighborhoods.predictor import ClassifierPredictor, RegressorPredictor


class NearestNeighbor:
    def __init__(self, type: str = "classifier"):
        if type not in ["classifier", "regressor"]:
            raise ValueError("type must be either 'classifier' or 'regressor'")
        
        self.predictor = ClassifierPredictor() if type == "classifier" else RegressorPredictor()
        self.search_algorithm = None
        self.metric = None
        self.k = None

    def compile(self, k: int, metrics: str = "euclidean", algorithm: str = "brute-force"):
        metric_map = {
            "euclidean": Euclidean(),
            "manhattan": Manhattan(),
            "chebyshev": Chebyshev(),
            "minkowski": Minkowski(),
            "cosine": Cosine(),
            "jaccard": Jaccard(),
            "hamming": Hamming()
        }
        
        if isinstance(metrics, str):
            if metrics not in metric_map:
                raise ValueError(f"Unknown metric: {metrics}")
            self.metric = metric_map[metrics]

        elif isinstance(metrics, DistanceMetric):
            self.metric = metrics

        else:
            raise ValueError("Metrics must be a string or a DistanceMetric instance")
        

        if algorithm == "brute-force":
            self.search_algorithm = BruteForceSearch()

        elif algorithm == "kd-tree":
            self.search_algorithm = KDTreeSearch()

        else:
            raise ValueError("Algorithm must be either 'brute-force' or 'kd-tree'")
        
        self.k = k

    def train(self, x_train: np.ndarray, y_train: np.ndarray):
        self.search_algorithm.train(x_train, y_train)

    def predict(self, x: np.ndarray) -> Union[int, float, np.ndarray]:
        if x.ndim > 1 and x.shape[0] > 1:
            return np.array([self.predict(x[i]) for i in range(x.shape[0])])
        
        neighbors = self.search_algorithm.find_neighbors(x, self.k, self.metric)
        return self.predictor.predict(neighbors)
