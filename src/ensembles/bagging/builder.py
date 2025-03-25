import sys
from typing import Optional, Literal, Union, Callable

from path_handler import PathManager

path_manager = PathManager()
sys.path.append(str(path_manager.get_base_directory()))

from src.ensembles.bagging.base import Bagging
from src.ensembles.bagging.components.estimator import Estimator
from src.ensembles.bagging.components.aggregator import PredictionAggregator


class BaggingBuilder:
    def __init__(self, aggregator: PredictionAggregator, error_metric: Callable):
        self.aggregator = aggregator
        self.error_metric = error_metric
        
        self.max_samples: Optional[int] = None
        self.n_estimators: Optional[int] = None
        self.estimator: Optional[Estimator] = None
        self.max_features: Optional[Union[int, Literal["sqrt", "log"]]] = None

    def compile(
        self, 
        estimator: Estimator, 
        n_estimators: int = 10, 
        max_samples: Optional[int] = None,
        max_features: Optional[Union[int, Literal["sqrt", "log"]]] = None,
    ) -> 'BaggingBuilder':

        self.estimator = estimator
        self.max_samples = max_samples
        self.max_features = max_features
        self.n_estimators = n_estimators
        
        return self
    
    def build(self) -> Bagging:
        if self.estimator is None:
            raise ValueError("The model is not compiled yet. Please call the compile method before fit.")

        return Bagging(
            max_samples=self.max_samples, 
            n_estimators=self.n_estimators, 
            estimator=self.estimator, 
            error_metric=self.error_metric,
            aggregator=self.aggregator, 
            max_features=self.max_features
        )