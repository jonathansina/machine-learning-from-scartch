import numpy as np
from TreeModels.TreePackage import IdentificationTree
from LinearModels.Losses import *
from matplotlib import pyplot as plt

class GradientBoostedRegressionTree(object):
    """
    A class for Gradient Boosted Regression Trees.

    Parameters
    ----------
    number_of_estimators : int, optional (default=10)
        The number of estimators in the ensemble.

    max_depth : int, optional (default=3)
        The maximum tree depth.

    max_features : int or str, optional (default="sqrt")
        The number of features to consider when looking for the best split.
        If "sqrt", then sqrt(number_of_features) are considered.

    loss : str, optional (default="squared-loss")
        The loss function to be used for optimization.
        Can be "squared-loss", "absolute-loss", or "huber".

    learning_rate : float, optional (default=0.1)
        The learning rate for each iteration.

    Attributes
    ----------
    y_mean : float
        The mean of the target variable.

    learning_rate : float
        The learning rate used in training.

    loss : LossFunction
        The loss function used in training.

    max_features : int or str
        The maximum number of features used in tree pruning.

    max_depth : int
        The maximum tree depth used in training.

    number_of_estimators : int
        The number of estimators in the ensemble.

    forest : list of IdentificationTree
        The list of trees in the ensemble.

    Methods
    -------
    compile(self, number_of_estimators=10, max_depth=3, max_features="sqrt",
            loss="squared-loss", learning_rate=0.1)
        Compiles the model by setting the hyperparameters.

    train(self, x_train, y_train, verbose=2)
        Trains the model on the given training data.

    predict(self, x)
        Predicts the target values for the given data.

    residual_error(self, pred, y)
        Calculates the residual error for a given prediction and target values.

    plot_loss(self, errors)
        Plots the loss function over the number of iterations.

    """

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
        """
        Compiles the model by setting the hyperparameters.

        Parameters
        ----------
        number_of_estimators : int, optional (default=10)
            The number of estimators in the ensemble.

        max_depth : int, optional (default=3)
            The maximum tree depth.

        max_features : int or str, optional (default="sqrt")
            The number of features to consider when looking for the best split.
            If "sqrt", then sqrt(number_of_features) are considered.

        loss : str, optional (default="squared-loss")
            The loss function to be used for optimization.
            Can be "squared-loss", "absolute-loss", or "huber".

        learning_rate : float, optional (default=0.1)
            The learning rate for each iteration.

        """
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
        """
        Trains the model on the given training data.

        Parameters
        ----------
        x_train : np.ndarray
            The training data features.

        y_train : np.ndarray
            The training data target values.

        verbose : int, optional (default=2)
            The verbosity level.

        Returns
        -------
        np.ndarray
            The list of errors for each iteration.

        """
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
        """
        Predicts the target values for the given data.

        Parameters
        ----------
        x : np.ndarray
            The data features.

        Returns
        -------
        np.ndarray
            The predicted target values.

        """
        prediction = np.array([self.y_mean] * len(x))
        for tree in self.forest:
            prediction += np.array(tree.predict(x)) * self.learning_rate
        return prediction

    def residual_error(self, pred, y: np.ndarray) -> np.ndarray:
        """
        Calculates the residual error for a given prediction and target values.

        Parameters
        ----------
        pred : np.ndarray
            The predicted target values.

        y : np.ndarray
            The target values.

        Returns
        -------
        np.ndarray
            The residual error.

        """
        return self.loss.derivative(pred, y)

    def plot_loss(self, errors: np.ndarray):
        """
        Plots the loss function over the number of iterations.

        Parameters
        ----------
        errors : np.ndarray
            The list of errors for each iteration.

        """
        plt.plot(range(self.number_of_estimators), errors)
        plt.xlabel('Number of Epochs')
        plt.ylabel('Cost')
        plt.title('Cost Function')
        plt.grid(True)
        plt.show()