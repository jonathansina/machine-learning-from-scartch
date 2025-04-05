import sys
from typing import Any, Literal, Tuple, Union, Optional

import numpy as np

from path_handler import PathManager

path_manager = PathManager()
sys.path.append(str(path_manager.get_base_directory()))

from src.tree.components.node import Node
from src.tree.components.strategy import TreeBuilderStrategy
from src.tree.components.utils import TreeUtils, FeatureUtils
from src.tree.components.impurity.base import ImpurityMeasure


class IdentificationTree:
    def __init__(
        self, 
        max_depth: int, 
        impurity_type: ImpurityMeasure, 
        builder_strategy: TreeBuilderStrategy, 
        max_features: Optional[Union[int, Literal["log", "sqrt"]]]
    ):

        self.max_depth = max_depth
        self.max_features = max_features
        self.impurity_measure = impurity_type
        self.builder_strategy = builder_strategy
        
        self.depth = 0
        self.root: Optional[Node] = None
        self.features_count: Optional[int] = None
        self.selected_features: Optional[int] = None

    def _build_tree(self, data: np.ndarray, depth: int) -> Node:
        is_single_class = len(np.unique(data[:, -1])) == 1
        has_repeated_features = (
            len(data) >= 2 and 
            (data[0, :-1] == data[1, :-1]).all() and 
            (data[0, -1] != data[1, -1]).all()
        )
        
        if is_single_class or has_repeated_features or depth >= self.max_depth:
            leaf_value = self.builder_strategy.determine_leaf_value(data)
            if self.depth < depth:
                self.depth = depth

            return Node(value=leaf_value, depth=depth)

        best_dim, best_threshold = self._find_best_split(data)
        left_indices, right_indices = TreeUtils.make_split(data[:, best_dim], best_threshold)

        node = Node(dimension=best_dim, threshold=best_threshold, depth=depth)
        node.left_child = self._build_tree(data[left_indices, :], depth + 1)
        node.right_child = self._build_tree(data[right_indices, :], depth + 1)
        return node

    def _find_best_split(self, data: np.ndarray) -> Tuple[int, float]:
        best_gain = float('-inf')
        best_dimension, best_threshold = None, None
        
        for dim in self.selected_features:
            thresholds = np.unique(data[:, dim])
            for threshold in thresholds:
                left_indices, right_indices = TreeUtils.make_split(data[:, dim], threshold)
    
                if len(left_indices) == 0 or len(right_indices) == 0:
                    continue
                    
                gain = self.impurity_measure.information_gain(data, left_indices, right_indices)
                if gain > best_gain:
                    best_gain = gain
                    best_dimension = dim
                    best_threshold = threshold
                    
        return best_dimension, best_threshold

    def _traverse(self, x: np.ndarray, node: Node) -> Any:
        if node.is_leaf():
            return node.value

        if x[node.dimension] <= node.threshold:
            return self._traverse(x, node.left_child)
        else:
            return self._traverse(x, node.right_child)

    def fit(self, x_train: np.ndarray, y_train: np.ndarray):
        if self.impurity_measure is None:
            raise ValueError("The model is not compiled yet. Please call the compile method before fit.")

        self.features_count = x_train.shape[1]
        n_features_to_use = FeatureUtils.get_feature_count(self.max_features, self.features_count)
        self.selected_features = np.random.choice(self.features_count, n_features_to_use, replace=False)
        
        data = np.column_stack((x_train, y_train))
        self.root = self._build_tree(data, 0)

    def predict(self, x: np.ndarray) -> Union[np.ndarray, Any]:
        if self.root is None:
            raise ValueError("The model is not trained yet. Please call the fit method before predict")

        if len(x.shape) > 1 and x.shape[0] > 1:
            return np.array([self._traverse(xi, self.root) for xi in x])

        return self._traverse(x, self.root)
    
    