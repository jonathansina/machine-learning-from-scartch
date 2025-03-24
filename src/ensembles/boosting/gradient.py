import sys
from typing import Literal, Optional, Union, List, Tuple

import numpy as np
from matplotlib import pyplot as plt

from path_handler import PathManager

path_manager = PathManager()
sys.path.append(str(path_manager.get_base_directory()))

from src.tree.factory import IdentificationTreeFactory, IdentificationTree
from src.tree.impurity.regression import MeanSquaredError, MeanAbsoluteError, Huber, ImpurityMeasure


class GradientBoostedRegressionTree:
    def __init__(self):
        self.y_mean = 0
        self.errors: List[float] = []
        self.max_depth: Optional[int] = None
        self.n_estimators: Optional[int] = None
        self.forest: List[IdentificationTree] = []
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

    def fit(self, x_train: np.ndarray, y_train: np.ndarray, verbose: int = 1):
        if self.impurity_type is None:
            raise ValueError("The model is not compiled yet. Please call the compile method before fit.")

        if y_train.ndim == 1:
            y_train = y_train.reshape(-1, 1)
    
        self.y_mean = np.mean(y_train)
        prediction = (np.ones(y_train.shape[0]) * self.y_mean).reshape(-1, 1)

        for i in range(self.n_estimators):
            residual = self.residual_error(prediction, y_train).reshape(-1, 1)

            tree = IdentificationTreeFactory.create("regressor")
            tree.compile(
                impurity_type=self.impurity_type[0],
                max_depth=self.max_depth,
                max_features=self.max_features
            )
            tree.fit(x_train, residual)

            predict = tree.predict(x_train).reshape(-1, 1)
            prediction += predict * self.learning_rate
            self.errors.append(self.impurity_type[1].calculate(prediction, y_train))
            self.forest.append(tree)

            if verbose >= 1:
                print(f"-------------------[Iteration {i}/{self.n_estimators}]---------------------")
                print('Error :', self.errors[-1])

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.forest is None:
            raise ValueError("The model is not trained yet. Please call the fit method before predict.")

        prediction = np.array([self.y_mean] * len(x))
        for tree in self.forest:
            prediction += np.array(tree.predict(x)) * self.learning_rate

        return prediction

    def residual_error(self, pred: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.impurity_type[1].derivative(pred, y)

    def plot_loss(self):
        plt.plot(range(self.n_estimators), self.errors)
        plt.xlabel('Number of Epochs')
        plt.ylabel('Cost')
        plt.title('Cost Function')
        plt.grid(True)
        plt.show()