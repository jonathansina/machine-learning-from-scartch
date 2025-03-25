import sys
from typing import Literal, Optional, Union, Tuple


from path_handler import PathManager

path_manager = PathManager()
sys.path.append(str(path_manager.get_base_directory()))

from src.ensembles.boosting.models.gradient.base import BoostingRegressor
from src.tree.components.impurity.regression import MeanSquaredError, MeanAbsoluteError, Huber, ImpurityMeasure


class BoostingRegressorBuilder:
    def __init__(self):
        self.max_depth: Optional[int] = None
        self.n_estimators: Optional[int] = None
        self.learning_rate: Optional[float] = None
        self.impurity_type: Optional[Tuple[str, ImpurityMeasure]] = None
        self.max_features: Optional[Union[int, Literal["sqrt", "log"]]] = None

    def _set_impurity_function(self, impurity_type: str):
        if impurity_type == "squared_loss":
            self.impurity_type = ("mse", MeanSquaredError())

        elif impurity_type == "absolute_loss":
            self.impurity_type = ("mae", MeanAbsoluteError())

        elif impurity_type == "huber":
            self.impurity_type = ("huber", Huber())

        else:
            raise ValueError(f"Unknown impurity measure: {impurity_type}")

    def compile(
        self, 
        max_depth: int = 3,
        n_estimators: int = 10, 
        learning_rate: int = 0.1,
        max_features: Optional[Union[int, Literal["sqrt", "log"]]] = None,
        impurity_type: Literal["squared_loss", "absolute_loss", "huber"] = "squared_loss"
    ):
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.learning_rate = learning_rate
        
        self._set_impurity_function(impurity_type)
        
        return self

    def build(self) -> BoostingRegressor:
        if self.impurity_type is None:
            raise ValueError("The model is not compiled yet. Please call the compile method before fit.")

        return BoostingRegressor(
            max_depth=self.max_depth, 
            n_estimators=self.n_estimators, 
            max_features=self.max_features,
            impurity_type=self.impurity_type,
            learning_rate=self.learning_rate
       )
  