import sys
import copy
from abc import ABC, abstractmethod
from typing import Optional, Literal, List, Union

import numpy as np
from sklearn.metrics import r2_score, accuracy_score

from path_handler import PathManager

path_manager = PathManager()
sys.path.append(str(path_manager.get_base_directory()))

from src.ensembles.bagging.utils import FeatureUtils
from src.ensembles.bagging.bootstrap import Bootstrap
from src.ensembles.bagging.estimator import Estimator
from src.ensembles.bagging.aggregator import ClassifierAggregator, RegressorAggregator, PredictionAggregator


class BaseBagging(ABC):
    def __init__(self):
        self.max_samples: Optional[int] = None
        self.n_estimators: Optional[int] = None
        self.estimator: Optional[Estimator] = None
        self.estimator_stack: List[Estimator] = []
        self.aggregator: Optional[PredictionAggregator]  = None
        self.max_features: Optional[Union[int, Literal["sqrt", "log"]]] = None

    def compile(
        self, 
        estimator: Estimator, 
        n_estimators: int = 10, 
        max_samples: Optional[int] = None,
        max_features: Optional[Union[int, Literal["sqrt", "log"]]] = None,
    ):
        self.estimator = estimator
        self.max_samples = max_samples
        self.max_features = max_features
        self.n_estimators = n_estimators

    def fit(self, x_train: np.ndarray, y_train: np.ndarray, verbose: int = 2, **kwargs):
        if self.estimator is None:
            raise ValueError("The model is not compiled yet. Please call the compile method before fit.")

        self.max_features = FeatureUtils.get_feature_count(self.max_features, x_train.shape[1])
        
        x_samples, y_samples = Bootstrap.create_samples(
            x_train, 
            y_train, 
            self.n_estimators, 
            self.max_samples
        )
        
        for i in range(self.n_estimators):
            estimator = copy.deepcopy(self.estimator)
            estimator.fit(x_samples[i], y_samples[i], **kwargs)
            self.estimator_stack.append(estimator)
            
            if verbose >= 2:
                print(f"---------------------- Learner {i + 1}/{self.n_estimators} Learned ----------------------")
                if verbose >= 1:
                    y_pred = self.estimator_stack[i].predict(x_samples[i])
                    training_error = self._calculate_error(y_samples[i], y_pred)
                    print(f"Training Error: {training_error}")

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

    @abstractmethod
    def _calculate_error(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        raise NotImplementedError("Subclasses must implement this method")


class BaggingClassifier(BaseBagging):
    def __init__(self):
        super().__init__()
        self.aggregator = ClassifierAggregator()
    
    def _calculate_error(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return accuracy_score(y_true, y_pred)


class BaggingRegressor(BaseBagging):
    def __init__(self):
        super().__init__()
        self.aggregator = RegressorAggregator()
    
    def _calculate_error(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return r2_score(y_true, y_pred)