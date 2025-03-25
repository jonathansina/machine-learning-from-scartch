import sys
from typing import Union, Optional, Literal

import numpy as np

from path_handler import PathManager

path_manager = PathManager()
sys.path.append(str(path_manager.get_base_directory()))

from src.neighborhood.components.predictor import Predictor
from src.neighborhood.components.search import SearchAlgorithm
from src.neighborhood.components.distances import DistanceMetric


class NearestNeighbor:
    def __init__(self, predictor: Predictor, k: int, metric: DistanceMetric, search_algorithm: SearchAlgorithm):
        self.k = k
        self.metric = metric
        self.predictor = predictor
        self.search_algorithm = search_algorithm

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
