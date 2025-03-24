import sys
from typing import Union, Optional, Literal

import numpy as np

from path_handler import PathManager

path_manager = PathManager()
sys.path.append(str(path_manager.get_base_directory()))

from src.neighborhood.distances import (
    Cosine, 
    Hamming,
    Jaccard, 
    Manhattan, 
    Chebyshev,
    Euclidean, 
    Minkowski,
    DistanceMetric
)
from src.neighborhood.predictor import Predictor
from src.neighborhood.search import BruteForceSearch, KDTreeSearch, SearchAlgorithm


class NearestNeighbor:
    def __init__(self, predictor: Predictor):
        self.predictor = predictor
        self.k: Optional[int] = None
        self.metric: Optional[DistanceMetric] = None
        self.search_algorithm: Optional[SearchAlgorithm] = None
        
    def _set_distance_metrics(self, metrics: str):
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
        
    def _set_algorithm(self, algorithm: str):
        if algorithm == "brute_force":
            self.search_algorithm = BruteForceSearch()

        elif algorithm == "kd_tree":
            self.search_algorithm = KDTreeSearch()

        else:
            raise ValueError("Algorithm must be either 'brute_force' or 'kd_tree'")

    def compile(
        self, k: int, 
        algorithm: Literal["brute_force", "kd_tree"] = "brute_force",
        metrics: Union[DistanceMetric, Literal["euclidean", "manhattan", "minkowski", "chebyshev", "cosine", "jaccard"]] = "euclidean"
    ):
        self.k = k
        self._set_algorithm(algorithm)
        self._set_distance_metrics(metrics)

    def fit(self, x_train: np.ndarray, y_train: np.ndarray):
        if self.k is None:
            raise ValueError("The model is not compiled yet. Please call the compile method before fit.")

        self.search_algorithm.fit(x_train, y_train)

    def predict(self, x: np.ndarray) -> Union[int, float, np.ndarray]:
        if self.search_algorithm.fitted is False:
            raise ValueError("The model is not trained yet. Please call the fit method before predict.")

        if x.ndim > 1 and x.shape[0] > 1:
            return np.array([self.predict(x[i]) for i in range(x.shape[0])])
        
        neighbors = self.search_algorithm.find_neighbors(x, self.k, self.metric)
        return self.predictor.predict(neighbors)
