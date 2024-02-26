import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern
from scipy.stats import norm


class POI(object):
    def __init__(self):
        """
        It estimates the probability that a point will improve upon the current best value. It considers the difference
        between the mean prediction and the current best value, taking into account the uncertainty in the surrogate
        model.
        """
        self.name = "probability_of_improvement"

    def __call__(self, y_pred: np.ndarray, y_std: np.ndarray, best_y: np.ndarray) -> np.ndarray:
        z = (y_pred - best_y) / y_std
        pi = norm.cdf(z)
        return pi


class EI(object):
    def __init__(self):
        """
        It selects points that have the potential to improve upon the best-observed value. It quantifies the expected
        improvement over the current best value and considers both the mean prediction of the surrogate model and its
        uncertainty.
        """
        self.name = "expected_improvement "

    def __call__(self, y_pred: np.ndarray, y_std: np.ndarray, best_y: np.ndarray) -> np.ndarray:
        z = (y_pred - best_y) / y_std
        ei = (y_pred - best_y) * norm.cdf(z) + y_std * norm.pdf(z)
        return ei


class UCB(object):
    def __init__(self, beta: int = 2):
        """
        It trades off exploration and exploitation by balancing the mean prediction of the surrogate model and an
        exploration term proportional to the uncertainty. It selects points that offer a good balance between predicted
        high values and exploration of uncertain regions.
        """
        self.name = "upper_confidence_bound"
        self.beta = beta

    def __call__(self, y_pred: np.ndarray, y_std: np.ndarray) -> np.ndarray:
        ucb = y_pred + self.beta * y_std
        return ucb


class BayesianTuning:
    """
    A class for performing Bayesian hyperparameter tuning using Gaussian Processes and acquisition functions.

    Parameters
    ----------
    acquisition : str, optional
        The acquisition function to be used for selecting the next point to evaluate, by default "ucb"
    kernel : str, optional
        The kernel function to be used in the Gaussian process, by default "rbf"

    Attributes
    ----------
    result : pd.DataFrame
        A dataframe containing the results of the tuning process, including the values of the objective function and the
        parameters at each iteration
    gp_model : sklearn.gaussian_process.GaussianProcessRegressor
        The Gaussian process model used for predicting the objective function
    kernel : sklearn.gaussian_process.kernels.Kernel
        The kernel function used in the Gaussian process
    acquisition : object
        The acquisition function used for selecting the next point to evaluate

    Methods
    -------
    compile(self, acquisition: str = "ucb", kernel: str = "rbf")
        Compiles the acquisition and kernel functions to be used in the tuning process
    train(self, objective_function, parameters_bound: dict, epochs: int, initial_points: int = 5) -> pd.DataFrame
        Trains the Gaussian process model and performs the tuning process, returning a dataframe containing the results
    get_result(self, indices: list, epoch_list: list, x_train: np.ndarray, y_train: np.ndarray, size_x: int)
        Returns the results of the tuning process as a dataframe
    """

    def __init__(self):
        self.result = None
        self.gp_model = None
        self.kernel = None
        self.acquisition = None

    def compile(self, acquisition: str = "ucb", kernel: str = "rbf"):
        """
        Compiles the acquisition and kernel functions to be used in the tuning process.

        Parameters
        ----------
        acquisition : str, optional
            The acquisition function to be used for selecting the next point to evaluate, by default "ucb"
        kernel : str, optional
            The kernel function to be used in the Gaussian process, by default "rbf"
        """
        if acquisition == "ucb":
            self.acquisition = UCB()
        elif acquisition == "poi":
            self.acquisition = POI()
        elif acquisition == "ei":
            self.acquisition = EI()
        elif isinstance(acquisition, (UCB, POI, EI)):
            self.acquisition = acquisition
        else:
            raise ValueError("Invalid acquisition function!")

        if kernel == "rbf":
            self.kernel = RBF()
        elif kernel == "matern":
            self.kernel = Matern()
        else:
            self.kernel = kernel
        self.gp_model = GaussianProcessRegressor(kernel=self.kernel)

    def train(self, objective_function, parameters_bound: dict, epochs: int, initial_points: int = 5) -> pd.DataFrame:
        """
        Trains the Gaussian process model and performs the tuning process, returning a dataframe containing the results.

        Parameters
        ----------
        objective_function : function
            The objective function to be optimized, taking the parameters as input and returning the objective value
        parameters_bound : dict
            A dictionary containing the bounds on the parameters, where the keys are the parameter names and the values are
            a tuple containing the minimum and maximum values
        epochs : int
            The number of iterations to be performed in the tuning process
        initial_points : int, optional
            The number of initial points to be sampled before the tuning process begins, by default 5

        Returns
        -------
        pd.DataFrame
            A dataframe containing the results of the tuning process, including the values of the objective function and the
            parameters at each iteration
        """
        indices = ["epochs", "result"] + list(parameters_bound.keys())
        epoch_list = [i for i in range(1, epochs + initial_points + 1)]

        x_train = []
        y_train = []
        size_x = len(parameters_bound.values())
        for _ in range(initial_points):
            x = []
            for i in parameters_bound:
                x.append(np.random.uniform(parameters_bound[i][0], parameters_bound[i][1]))
            x_train.append(x)
            y_train.append([objective_function(*x)])

        x_train = np.array(x_train).reshape(-1, size_x)
        y_train = np.array(y_train).reshape(-1, 1)

        x_range = []
        for i in parameters_bound:
            x_range.append(np.linspace(parameters_bound[i][0] + 1e-9, parameters_bound[i][1], 100))
        x_range = np.array(x_range).T

        counter = 0
        while counter < epochs:
            counter += 1
            # Fit the Gaussian process model to the sampled points
            self.gp_model.fit(x_train, y_train)

            # Generate predictions using the Gaussian process model
            y_pred, y_std = self.gp_model.predict(x_range, return_std=True)

            new_x_train = x_range[np.argmax(self.acquisition(y_pred, y_std))].reshape(-1, size_x)
            new_y_train = objective_function(*new_x_train).reshape(-1, 1)

            x_train = np.append(x_train, new_x_train)
            y_train = np.append(y_train, new_y_train)

        return self.get_result(indices, epoch_list, x_train, y_train, size_x)

    def get_result(self, indices: list, epoch_list: list, x_train: np.ndarray, y_train: np.ndarray, size_x: int):
        """
        Returns the results of the tuning process as a dataframe.

        Parameters
        ----------
        indices : list
            A list containing the column names to be included in the dataframe
        epoch_list : list
            A list containing the iteration numbers
        x_train : np.ndarray
            The array of sampled parameters
        y_train : np.ndarray
            The array of objective function values
        size_x : int
            The number of parameters

        Returns
        -------
        pd.DataFrame
            A dataframe containing the results of the tuning process
        """
        data = np.column_stack((epoch_list, y_train, x_train.reshape(-1, size_x)))
        self.result = pd.DataFrame(data=data, columns=indices)
        return self.result
