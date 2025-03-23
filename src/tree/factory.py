import sys
from typing import Optional, Union, Literal

from path_handler import PathManager

path_manager = PathManager()
sys.path.append(str(path_manager.get_base_directory()))

from src.tree.base import IdentificationTree
from src.tree.impurity.classification import Gini, Entropy
from src.tree.strategy import ClassificationTreeBuilder, RegressionTreeBuilder
from src.tree.impurity.regression import MeanSquaredError, MeanAbsoluteError, Huber


class IdentificationTreeFactory:
    @staticmethod
    def create(strategy_type: Literal["classifier", "regressor"], ) -> IdentificationTree:
        if strategy_type == "classifier":
            builder_strategy = ClassificationTreeBuilder()
        
        elif strategy_type == "regressor":
            builder_strategy = RegressionTreeBuilder()
        
        else:
            raise ValueError(f"Unknown strategy type: {strategy_type}")
        
        return IdentificationTree(builder_strategy=builder_strategy)