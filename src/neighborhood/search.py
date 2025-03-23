import sys
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import numpy as np

from path_handler import PathManager

path_manager = PathManager()
sys.path.append(str(path_manager.get_base_directory()))

from src.neighborhood.tree import KDTree, KDNode
from src.neighborhood.distances import DistanceMetric


class SearchAlgorithm(ABC):
    @abstractmethod
    def train(self, x_train: np.ndarray, y_train: np.ndarray):
        raise NotImplementedError("Subclasses must implement this method")
    
    @abstractmethod
    def find_neighbors(self, x: np.ndarray, k: int, metric: DistanceMetric) -> List:
        raise NotImplementedError("Subclasses must implement this method")


class BruteForceSearch(SearchAlgorithm):
    def __init__(self):
        self.x_train: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
    
    def train(self, x_train: np.ndarray, y_train: np.ndarray):
        self.x_train = x_train
        self.y_train = y_train
    
    def find_neighbors(self, x: np.ndarray, k: int, metric: DistanceMetric) -> np.ndarray:
        distances = [
            metric(self.x_train[i], x) 
            for i in range(self.x_train.shape[0])
        ]

        indices = np.argsort(distances)[:k]
        return self.y_train[indices]


class KDTreeSearch(SearchAlgorithm):
    def __init__(self):
        self.tree: Optional[KDTree] = None
        self.knn_set: List[Tuple[KDNode, float]] = []
    
    def train(self, x_train: np.ndarray, y_train: np.ndarray):
        self.tree = KDTree(values=x_train, labels=y_train)
    
    def find_neighbors(self, x: np.ndarray, k: int, metric: DistanceMetric) -> List:
        self.knn_set: List[Tuple[KDNode, float]] = []
        self._search_tree(x, self.tree.root, k, metric)
        return [
            node.label 
            for node, _ in self.knn_set
        ]
    
    def _search_tree(self, x: np.ndarray, node: KDNode, k: int, metric: DistanceMetric):
        if node is None:
            return
            
        distance = metric(x, node.value)
        
        duplicate = [
            metric(node.value, item[0].value) < 1e-4 
            for item in self.knn_set
        ]
        
        if not np.array(duplicate, bool).any():
            if len(self.knn_set) < k:
                self.knn_set.append((node, distance))

            elif distance < self.knn_set[0][1]:
                self.knn_set[0] = (node, distance)
        
        self.knn_set = sorted(self.knn_set, key=lambda x: -x[1])
        
        current_dimension = node.depth % self.tree.dimensions
        if len(self.knn_set) < k or abs(x[current_dimension] - node.value[current_dimension]) < self.knn_set[0][1]:
            self._search_tree(x, node.lchild, k, metric)
            self._search_tree(x, node.rchild, k, metric)

        elif x[current_dimension] < node.value[current_dimension]:
            self._search_tree(x, node.lchild, k, metric)

        else:
            self._search_tree(x, node.rchild, k, metric)