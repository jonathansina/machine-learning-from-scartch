import sys
from typing import Optional, Callable, Tuple, Dict, Any

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor

from path_handler import PathManager

path_manager = PathManager()
sys.path.append(str(path_manager.get_base_directory()))

from src.tunning.utils import KernelUtils
from src.tunning.result import OptimizationResult
from src.tunning.asquisition.base import POI, EI, AcquisitionFunction
from src.tunning.asquisition.factory import AcquisitionFunctionFactory


class BayesianTuning:
    def __init__(self):
        self.result_handler = OptimizationResult()
        self.acquisition: Optional[AcquisitionFunction] = None
        self.gp_model: Optional[GaussianProcessRegressor] = None

    def compile(self, acquisition: str = "ucb", kernel: str = "rbf", **kwargs):
        kernel_obj = KernelUtils.create(kernel)
        self.gp_model = GaussianProcessRegressor(kernel=kernel_obj)
        self.acquisition = AcquisitionFunctionFactory.create(acquisition, **kwargs)

    def fit(
        self, 
        objective_function: Callable, 
        parameters_bound: Dict[str, Any], 
        epochs: int, 
        initial_points: int = 5
    ) -> pd.DataFrame:

        indices = ["epochs", "result"] + list(parameters_bound.keys())
        epoch_list = [i for i in range(1, epochs + initial_points + 1)]

        x_train, y_train, size_x = self._initialize_samples(objective_function, parameters_bound, initial_points)

        x_range = self._generate_parameter_grid(parameters_bound)
        
        for _ in range(epochs):
            self.gp_model.fit(x_train, y_train)
            y_pred, y_std = self.gp_model.predict(x_range, return_std=True)
            
            if isinstance(self.acquisition, (POI, EI)):
                best_y = np.max(y_train)
                acq_values = self.acquisition(y_pred, y_std, best_y=best_y)

            else:
                acq_values = self.acquisition(y_pred, y_std)

            new_x_train = x_range[np.argmax(acq_values)].reshape(1, size_x)
            new_y_train = np.array([objective_function(*new_x_train[0])]).reshape(-1, 1)
            
            x_train = np.vstack((x_train, new_x_train))
            y_train = np.vstack((y_train, new_y_train))

        return self.result_handler.create_dataframe(indices, epoch_list, x_train, y_train, size_x)
    
    def _initialize_samples(
        self, 
        objective_function: Callable, 
        parameters_bound: Dict[str, Any], 
        initial_points: int
    ) -> Tuple[np.ndarray, np.ndarray, int]:

        x_train = []
        y_train = []
        size_x = len(parameters_bound)
        
        for _ in range(initial_points):
            x = []
            for param_name, bounds in parameters_bound.items():
                x.append(np.random.uniform(bounds[0], bounds[1]))
            x_train.append(x)
            y_train.append([objective_function(*x)])
            
        return np.array(x_train), np.array(y_train), size_x
    
    def _generate_parameter_grid(self, parameters_bound: Dict[str, Any], n_points: int = 100):
        x_range = []
        for param_name, bounds in parameters_bound.items():
            x_range.append(np.linspace(bounds[0] + 1e-9, bounds[1], n_points))

        return np.array(x_range).T