import sys

import numpy as np

from path_handler import PathManager

path_manager = PathManager()
sys.path.append(str(path_manager.get_base_directory()))

from src.linear.base import LinearModel
from src.linear.components.loss import LogLoss, BinaryCrossEntropy


class LogisticRegression(LinearModel):
    def _validate_loss_function(self, loss):
        if isinstance(loss, LogLoss) or isinstance(loss, BinaryCrossEntropy) or loss == 'log_loss' or loss == 'binary_crossentropy':
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
        
        y_out = 1 / (1 + np.exp(-(self.weight_matrix.dot(x) + self.bias)))
        return y_out

    def classify(self, x: np.ndarray, threshold: float = 0.5):
        predicted = self.predict(x)
        return np.array(
            [
                0 if p < threshold else 1 
                for p in predicted
            ]
        )