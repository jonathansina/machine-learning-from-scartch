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
    def create(
        strategy_type: Literal["classifier", "regressor"], 
        impurity_type: str, 
        max_depth: Optional[int] = 10, 
        max_features: Optional[Union[int, str]] = None
    ) -> IdentificationTree:

        if strategy_type == "classifier":
            return IdentificationTreeFactory.create_classifier(impurity_type, max_depth, max_features)
        
        elif strategy_type == "regressor":
            return IdentificationTreeFactory.create_regressor(impurity_type, max_depth, max_features)
        
        else:
            raise ValueError(f"Unknown strategy type: {strategy_type}")
        
    @staticmethod
    def create_classifier(impurity_type: str = "gini", max_depth: int = 10, max_features: Optional[Union[int, str]] = None) -> IdentificationTree:
        if impurity_type == "gini":
            impurity = Gini()

        elif impurity_type == "entropy":
            impurity = Entropy()

        else:
            raise ValueError(f"Unknown classification impurity type: {impurity_type}")
        
        return IdentificationTree(
            impurity_measure=impurity,
            builder_strategy=ClassificationTreeBuilder(),
            max_depth=max_depth,
            max_features=max_features
        )
    
    @staticmethod
    def create_regressor(impurity_type: str = "mse", max_depth: int = 10, max_features: Optional[Union[int, str]] = None) -> IdentificationTree:
        if impurity_type == "mse":
            impurity = MeanSquaredError()

        elif impurity_type == "mae":
            impurity = MeanAbsoluteError()

        elif impurity_type == "huber":
            impurity = Huber()

        else:
            raise ValueError(f"Unknown regression impurity type: {impurity_type}")
        
        return IdentificationTree(
            impurity_measure=impurity,
            builder_strategy=RegressionTreeBuilder(),
            max_depth=max_depth,
            max_features=max_features
        )