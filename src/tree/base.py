import sys
from typing import Any, Literal, Tuple, Union, Optional, Dict

import numpy as np

from path_handler import PathManager

path_manager = PathManager()
sys.path.append(str(path_manager.get_base_directory()))

from src.tree.node import Node
from src.tree.utils import TreeUtils
from src.tree.impurity.base import ImpurityMeasure
from src.tree.impurity.classification import Gini, Entropy
from src.tree.impurity.regression import MeanSquaredError, MeanAbsoluteError, Huber
from src.tree.strategy import TreeBuilderStrategy, ClassificationTreeBuilder, RegressionTreeBuilder


class IdentificationTree:
    def __init__(self, builder_strategy: TreeBuilderStrategy,):
        self.max_depth: Optional[int] = None
        self.builder_strategy = builder_strategy
        self.impurity_measure: Optional[ImpurityMeasure] = None
        self.max_features: Optional[Union[int, Literal["log", "sqrt"]]] = None
        
        self.depth = 0
        self.root: Optional[Node] = None
        self.features_count: Optional[int] = None
        self.selected_features: Optional[int] = None

    def _determine_max_features(self, n_features: int) -> int:
        if self.max_features is None:
            return n_features

        elif self.max_features == "sqrt":
            return int(np.sqrt(n_features)) + 1

        elif self.max_features == "log":
            return int(np.log(n_features)) + 1

        else:
            return self.max_features

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
    
    def compile(
        self, 
        impurity_type: Literal["gini", "entropy", "mse", "mae", "huber"], 
        max_depth: int = 10,
        max_features: Optional[Union[int, Literal["sqrt", "log"]]] = None
    ):
        self.max_depth = max_depth
        self.max_features = max_features
        
        classification_impurity: Dict[str, ] = {
            "gini": Gini(),
            "entropy": Entropy()
        }
        
        regression_impurity = {
            "mse": MeanSquaredError(),
            "mae": MeanAbsoluteError(),
            "huber": Huber()
        }

        if isinstance(self.builder_strategy, ClassificationTreeBuilder):
            if impurity_type in classification_impurity:
                self.impurity_measure = classification_impurity[impurity_type]
            
            else:
                raise ValueError(f"Unknown classification impurity measure: {impurity_type}")
        
        elif isinstance(self.builder_strategy, RegressionTreeBuilder):
            if impurity_type in regression_impurity:
                self.impurity_measure = regression_impurity[impurity_type]
            
            else:
                raise ValueError(f"Unknown regression impurity measure: {impurity_type}")

    def fit(self, x_train: np.ndarray, y_train: np.ndarray):
        if self.impurity_measure is None:
            raise ValueError("You must call compile() before training the model")

        self.features_count = x_train.shape[1]
        n_features_to_use = self._determine_max_features(self.features_count)
        self.selected_features = np.random.choice(self.features_count, n_features_to_use, replace=False)
        
        data = np.column_stack((x_train, y_train))
        self.root = self._build_tree(data, 0)

    def predict(self, x: np.ndarray) -> Union[np.ndarray, Any]:
        if len(x.shape) > 1 and x.shape[0] > 1:
            return np.array([self._traverse(xi, self.root) for xi in x])

        return self._traverse(x, self.root)