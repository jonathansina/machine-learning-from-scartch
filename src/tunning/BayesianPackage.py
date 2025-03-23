import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern
from scipy.stats import norm


class POI(object):
    def __init__(self):
        self.name = "probability_of_improvement"

    def __call__(self, y_pred: np.ndarray, y_std: np.ndarray, best_y: np.ndarray) -> np.ndarray:
        z = (y_pred - best_y) / y_std
        pi = norm.cdf(z)
        return pi


class EI(object):
    def __init__(self):
        self.name = "expected_improvement "

    def __call__(self, y_pred: np.ndarray, y_std: np.ndarray, best_y: np.ndarray) -> np.ndarray:
        z = (y_pred - best_y) / y_std
        ei = (y_pred - best_y) * norm.cdf(z) + y_std * norm.pdf(z)
        return ei


class UCB(object):
    def __init__(self, beta: int = 2):
        self.name = "upper_confidence_bound"
        self.beta = beta

    def __call__(self, y_pred: np.ndarray, y_std: np.ndarray) -> np.ndarray:
        ucb = y_pred + self.beta * y_std
        return ucb


class BayesianTuning:
    def __init__(self):
        self.result = None
        self.gp_model = None
        self.kernel = None
        self.acquisition = None

    def compile(self, acquisition: str = "ucb", kernel: str = "rbf"):
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
        data = np.column_stack((epoch_list, y_train, x_train.reshape(-1, size_x)))
        self.result = pd.DataFrame(data=data, columns=indices)
        return self.result
