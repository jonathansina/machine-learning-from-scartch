import sys
import copy
import concurrent.futures
from typing import Literal, List, Union, Callable, Optional

import numpy as np

from path_handler import PathManager

path_manager = PathManager()
sys.path.append(str(path_manager.get_base_directory()))

from src.ensembles.bagging.components.utils import FeatureUtils
from src.ensembles.bagging.components.bootstrap import Bootstrap
from src.ensembles.bagging.components.estimator import Estimator
from src.ensembles.bagging.components.aggregator import PredictionAggregator


class Bagging:
    def __init__(
        self, 
        n_estimators: int, 
        estimator: Estimator, 
        error_metric: Callable,
        max_samples: Optional[int], 
        aggregator: PredictionAggregator, 
        max_features: Optional[Union[int, Literal["sqrt", "log"]]]
    ):
        self.estimator = estimator
        self.aggregator = aggregator
        self.max_samples = max_samples
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.error_metric = error_metric
        self.estimator_stack: List[Estimator] = []
        
    def _fit_estimator(self, i: int, x_samples: np.ndarray, y_samples: np.ndarray, verbose: int) -> Estimator:
        estimator = copy.deepcopy(self.estimator)
        estimator.fit(x_samples[i], y_samples[i])
        
        if verbose >= 2:
            print(f"---------------------- Learner {i + 1}/{self.n_estimators} Learned ----------------------")

        if verbose >= 1:
            y_pred = estimator.predict(x_samples[i])
            training_error = self.error_metric(y_samples[i], y_pred)
            print(f"Training Error: {training_error}")

        return estimator

    def fit(self, x_train: np.ndarray, y_train: np.ndarray, n_jobs: int = 1, verbose: int = 2):
        if self.estimator is None:
            raise ValueError("The model is not compiled yet. Please call the compile method before fit.")

        self.max_features = FeatureUtils.get_feature_count(self.max_features, x_train.shape[1])
        
        x_samples, y_samples = Bootstrap.create_samples(
            x_train, 
            y_train, 
            self.n_estimators, 
            self.max_samples
        )

        with concurrent.futures.ThreadPoolExecutor(max_workers=n_jobs) as executor:
            results = list(
                executor.map(
                    lambda i: self._fit_estimator(i, x_samples, y_samples, verbose), 
                    range(self.n_estimators)
                )
            )

        self.estimator_stack.extend(results)

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.estimator_stack is None:
            raise ValueError("The model is not trained yet. Please call the fit method before predict.")

        if x.T.shape != x.shape and x.shape[0] != 1:
            return np.array(
                [
                    self.predict(x[i]) 
                    for i in range(x.shape[0])
                ]
            )
        
        predictions = [
            model.predict(x) 
            for model in self.estimator_stack
        ]
        return self.aggregator.aggregate(predictions)