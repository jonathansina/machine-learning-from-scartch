import sys

import numpy as np

from path_handler import PathManager

path_manager = PathManager()
sys.path.append(str(path_manager.get_base_directory()))

from src.linear.base import BaseLinearModel
from src.linear.components.loss import Hinge


class SVM(BaseLinearModel):
    def _validate_loss_function(self, loss):
        if loss == 'hinge' or isinstance(loss, Hinge):
            return None

        else:
            raise ValueError(f'Invalid loss function: {loss}')

    def predict(self, x: np.ndarray):
        if x.ndim > 1 and x.shape[0] > 1:
            return np.array(
                [
                    self.predict(x[i]) 
                    for i in range(x.shape[0])
                ]
            ).reshape(x.shape[0], 1)

        y_out = np.sign(self.weight_matrix.dot(x) + self.bias)
        return y_out