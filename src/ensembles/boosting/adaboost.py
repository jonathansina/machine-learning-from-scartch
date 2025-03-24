import sys
from typing import Literal, Optional, Union, List, Tuple

import numpy as np
from matplotlib import pyplot as plt

from path_handler import PathManager

path_manager = PathManager()
sys.path.append(str(path_manager.get_base_directory()))

from src.tree.factory import IdentificationTreeFactory, IdentificationTree
from src.ensembles.boosting.loss import ExponentialLoss, LogisticLoss, SquaredLoss, LossFunction


class AdaBoostClassifier:
    def __init__(self):
        self.errors: List[float] = []
        self.max_depth: Optional[int] = None
        self.loss_type: Optional[str] = None
        self.n_estimators: Optional[int] = None
        self.classes_: Optional[np.ndarray] = None
        self._loss_fn: Optional[LossFunction] = None
        self.forest: List[Tuple[IdentificationTree, float]] = []
        self.max_features: Optional[Union[int, Literal["sqrt", "log"]]] = None

    def compile(
        self, 
        impurity_type: Literal["gini", "entropy"],
        max_depth: int = 1,
        n_estimators: int = 50, 
        max_features: Optional[Union[int, Literal["sqrt", "log"]]] = None,
        loss: Literal["exponential", "logistic", "squared"] = "exponential"
    ):
        self.loss_type = loss
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.impurity_type = impurity_type
        
        self._set_loss_function(loss)
    
    def _set_loss_function(self, loss: str):
        if loss == "exponential":
            self._loss_fn = ExponentialLoss()

        elif loss == "logistic":
            self._loss_fn = LogisticLoss()

        elif loss == "squared":
            self._loss_fn = SquaredLoss()

        else:
            raise ValueError(f"Unsupported loss function: {loss}")
    

    def fit(self, x_train: np.ndarray, y_train: np.ndarray, verbose: int = 2):
        if self.impurity_type is None:
            raise ValueError("The model is not compiled yet. Please call the compile method before fit.")

        n_samples = x_train.shape[0]
        self.classes_ = np.unique(y_train)
        
        sample_weights = np.ones(n_samples) / n_samples
        
        for i in range(self.n_estimators):
            tree = IdentificationTreeFactory.create("classifier")
            tree.compile(
                impurity_type=self.impurity_type,
                max_depth=self.max_depth,
                max_features=self.max_features
            )
            
            tree.fit(x_train, y_train, sample_weights=sample_weights)
            predictions = tree.predict(x_train)

            incorrect = (predictions != y_train).astype(float)
            error = np.sum(sample_weights * incorrect) / np.sum(sample_weights)
            
            if error >= 0.5 or error == 0:
                if i == 0:
                    if error < 0.5:
                        estimator_weight = 1.0
                        self.forest.append((tree, estimator_weight))
                break

            estimator_weight, sample_weights = self._loss_fn.compute(error, incorrect, sample_weights)
            sample_weights /= np.sum(sample_weights)

            self.forest.append((tree, estimator_weight))
            
            ensemble_pred = self.predict(x_train)
            ensemble_error = np.mean(ensemble_pred != y_train)
            self.errors.append(ensemble_error)
            
            if verbose >= 1:
                print(f"-------------------[Iteration {i + 1}/{self.n_estimators}]---------------------")
                print(f'Estimator Error: {error:.4f}')
                print(f'Ensemble Error: {ensemble_error:.4f}')
                print(f'Estimator Weight: {estimator_weight:.4f}')
        
        if verbose >= 0:
            self.plot_loss()

    def predict(self, x: np.ndarray) -> np.ndarray:
        if len(self.forest) == 0:
            raise ValueError("The model is not trained yet. Please call the fit method before predict.")
            
        if len(self.classes_) == 2:
            return self._predict_binary(x)
        else:
            return self._predict_multiclass(x)

    def _predict_binary(self, x: np.ndarray) -> np.ndarray:
        n_samples = x.shape[0]
        scores = np.zeros(n_samples)
        
        for tree, weight in self.forest:
            predictions = tree.predict(x)
            adjusted_predictions = np.where(predictions == self.classes_[1], 1, -1)
            scores += weight * adjusted_predictions
        
        return np.where(scores >= 0, self.classes_[1], self.classes_[0])

    def _predict_multiclass(self, x: np.ndarray) -> np.ndarray:
        n_samples = x.shape[0]
        scores = np.zeros((n_samples, len(self.classes_)))
        
        for tree, weight in self.forest:
            predictions = tree.predict(x)
            for i, pred in enumerate(predictions):
                class_idx = np.where(self.classes_ == pred)[0][0]
                scores[i, class_idx] += weight
        
        return self.classes_[np.argmax(scores, axis=1)]

    def plot_loss(self):
        plt.plot(range(1, len(self.errors) + 1), self.errors)
        plt.xlabel('Number of Estimators')
        plt.ylabel('Error Rate')
        plt.title('AdaBoost Error Rate')
        plt.grid(True)
        plt.show()