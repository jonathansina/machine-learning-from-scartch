import sys
from typing import Literal, Optional, Union, List, Tuple

import numpy as np
from matplotlib import pyplot as plt

from path_handler import PathManager

path_manager = PathManager()
sys.path.append(str(path_manager.get_base_directory()))

from src.tree.base import IdentificationTree
from src.tree.factory import IdentificationTreeFactory 
from src.tree.components.impurity.regression import ImpurityMeasure


class BoostingRegressor:
    def __init__(
        self, 
        max_depth: int, 
        n_estimators: int, 
        learning_rate: float, 
        impurity_type: Tuple[str, ImpurityMeasure], 
        max_features: Optional[Union[int, Literal["sqrt", "log"]]]
    ):
        self.y_mean = 0
        self.errors: List[float] = []
        self.forest: List[IdentificationTree] = []
        
        self.max_depth = max_depth
        self.max_features = max_features
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.impurity_type = impurity_type

    def fit(self, x_train: np.ndarray, y_train: np.ndarray, verbose: int = 1):
        if self.impurity_type is None:
            raise ValueError("The model is not compiled yet. Please call the compile method before fit.")

        if y_train.ndim == 1:
            y_train = y_train.reshape(-1, 1)
    
        self.y_mean = np.mean(y_train)
        prediction = (np.ones(y_train.shape[0]) * self.y_mean).reshape(-1, 1)

        for i in range(self.n_estimators):
            residual = self.residual_error(prediction, y_train).reshape(-1, 1)

            tree = (
                IdentificationTreeFactory.create("regressor")
                .compile(
                    max_depth=self.max_depth,
                    max_features=self.max_features,
                    impurity_type=self.impurity_type[0]
                )
                .build()
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