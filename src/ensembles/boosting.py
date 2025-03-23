import numpy as np
from TreeModels.TreePackage import IdentificationTree
from LinearModels.Losses import *
from matplotlib import pyplot as plt

class GradientBoostedRegressionTree(object):
    def __init__(self):
        self.y_mean = 0
        self.learning_rate = None
        self.loss = None
        self.max_features = None
        self.max_depth = None
        self.number_of_estimators = None
        self.forest = []

    def compile(self, number_of_estimators: int = 10, max_depth: int = 3, max_features: int | str = "sqrt",
                loss: str = "squared-loss", learning_rate: int = 0.1):

        self.learning_rate = learning_rate
        self.number_of_estimators = number_of_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        if loss == "squared-loss":
            self.loss = MeanSquaredError()
        elif loss == "absolute-loss":
            self.loss = MeanAbsoluteError()
        elif loss == "huber":
            self.loss = Huber()
        else:
            raise ValueError("Invalid loss!")

    def train(self, x_train: np.ndarray, y_train: np.ndarray, verbose: int = 2) -> np.ndarray:
        self.y_mean = np.mean(y_train)
        prediction = (np.ones(y_train.shape[0]) * self.y_mean).reshape(-1, 1)
        errors = []
        for i in range(self.number_of_estimators):
            residual = self.residual_error(prediction, y_train).reshape(-1, 1)

            tree = IdentificationTree(type="regressor")
            tree.compile(
                max_depth=self.max_depth,
                max_features=self.max_features,
                impurity_function="mse"
            )
            tree.train(x_train, residual)

            predict = tree.predict(x_train).reshape(-1, 1)
            prediction += predict * self.learning_rate
            errors.append(self.loss(prediction, y_train))
            self.forest.append(tree)

            if verbose <= 1:
                print(f"-------------------[Iteration {i}/{self.number_of_estimators}]---------------------")
                print('Error :', errors[-1])

        if verbose <= 0:
            self.plot_loss(np.array(errors))
        return np.array(errors)

    def predict(self, x: np.ndarray) -> np.ndarray:
        prediction = np.array([self.y_mean] * len(x))
        for tree in self.forest:
            prediction += np.array(tree.predict(x)) * self.learning_rate
        return prediction

    def residual_error(self, pred, y: np.ndarray) -> np.ndarray:
        return self.loss.derivative(pred, y)

    def plot_loss(self, errors: np.ndarray):
        plt.plot(range(self.number_of_estimators), errors)
        plt.xlabel('Number of Epochs')
        plt.ylabel('Cost')
        plt.title('Cost Function')
        plt.grid(True)
        plt.show()