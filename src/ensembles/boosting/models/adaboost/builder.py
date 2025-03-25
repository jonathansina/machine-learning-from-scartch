import sys
from typing import Literal, Optional, Union

from path_handler import PathManager

path_manager = PathManager()
sys.path.append(str(path_manager.get_base_directory()))

from src.ensembles.boosting.models.adaboost.base import BoostingClassifier
from src.ensembles.boosting.components.loss import ExponentialLoss, LogisticLoss, SquaredLoss, LossFunction


class BoostingClassifierBuilder:
    def __init__(self):
        self.max_depth: Optional[int] = None
        self.n_estimators: Optional[int] = None
        self._loss_fn: Optional[LossFunction] = None
        self.impurity_type: Optional[Literal["gini", "entropy"]] = None
        self.max_features: Optional[Union[int, Literal["sqrt", "log"]]] = None
        
    def _set_loss_function(self, loss: str):
        if loss == "exponential":
            self._loss_fn = ExponentialLoss()

        elif loss == "logistic":
            self._loss_fn = LogisticLoss()

        elif loss == "squared":
            self._loss_fn = SquaredLoss()

        else:
            raise ValueError(f"Unsupported loss function: {loss}")

    def compile(
        self, 
        impurity_type: Literal["gini", "entropy"],
        max_depth: int = 1,
        n_estimators: int = 50, 
        max_features: Optional[Union[int, Literal["sqrt", "log"]]] = None,
        loss: Literal["exponential", "logistic", "squared"] = "exponential"
    ) -> "BoostingClassifierBuilder":

        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.impurity_type = impurity_type
        
        self._set_loss_function(loss)
        
        return self
    
    def build(self) -> BoostingClassifier:
        if self._loss_fn is None:
            raise ValueError("The model is not compiled yet. Please call the compile method before build.")
        
        return BoostingClassifier(
            loss_fn=self._loss_fn,
            max_depth=self.max_depth,
            max_features=self.max_features,
            n_estimators=self.n_estimators,
            impurity_type=self.impurity_type
        )
