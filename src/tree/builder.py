import sys
from typing import Literal, Union, Optional, Dict

from path_handler import PathManager

path_manager = PathManager()
sys.path.append(str(path_manager.get_base_directory()))

from src.tree.base import IdentificationTree
from src.tree.components.impurity.base import ImpurityMeasure
from src.tree.components.impurity.classification import Gini, Entropy
from src.tree.components.impurity.regression import MeanSquaredError, MeanAbsoluteError, Huber
from src.tree.components.strategy import TreeBuilderStrategy, ClassificationTreeBuilder, RegressionTreeBuilder


class IdentificationTreeBuilder:
    def __init__(self, builder_strategy: TreeBuilderStrategy):
        self.max_depth: Optional[int] = None
        self.builder_strategy = builder_strategy
        self.impurity_measure: Optional[ImpurityMeasure] = None
        self.max_features: Optional[Union[int, Literal["log", "sqrt"]]] = None
 
    def _set_impurity_function(self, impurity_type: str):
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

    def compile(
        self, 
        impurity_type: Literal["gini", "entropy", "mse", "mae", "huber"], 
        max_depth: int = 10,
        max_features: Optional[Union[int, Literal["sqrt", "log"]]] = None
    ):
        self.max_depth = max_depth
        self.max_features = max_features
        self._set_impurity_function(impurity_type)
        
        return self
    
    def build(self) -> IdentificationTree:
        if self.impurity_measure is None:
            raise ValueError("The model is not compiled yet. Please call the compile method before build.")
        
        return IdentificationTree(
            max_depth=self.max_depth, 
            max_features=self.max_features,
            impurity_type=self.impurity_measure, 
            builder_strategy=self.builder_strategy
        )
