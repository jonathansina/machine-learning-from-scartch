import sys
from typing import Literal, Optional, Union

import numpy as np

from path_handler import PathManager

path_manager = PathManager()
sys.path.append(str(path_manager.get_base_directory()))

from src.linear.components.loss import LossFunction
from src.linear.components.optimizer import Optimizer
from src.linear.components.regularizer import Regularizer
from src.linear.components.factory import ComponentFactory
from src.linear.models.logistic.base import LogisticRegression


class LogisticRegressionlBuilder:
    def __init__(self):
        self.bias = np.array([])
        self.loss: Optional[LossFunction] = None
        self.weight_matrix = np.array([])
        self.optimizer: Optional[Optimizer] = None
        self.regularizer: Optional[Regularizer] = None
        
    def compile(
        self, 
        optimizer: Union[Literal["adam", "adagrad", "rmsprop", "sgd", "newton_method"], Optimizer], 
        loss: Union[Literal["binary_crossentropy", "log_loss"], LossFunction], 
        regularizer: Optional[Union[Literal["l1", "l2", "l1l2"], Regularizer]] = None
    ) -> 'LogisticRegressionlBuilder':

        self.loss = ComponentFactory.create_loss(loss)
        self.optimizer = ComponentFactory.create_optimizer(optimizer)
        self.regularizer = ComponentFactory.create_regularizer(regularizer)
        
        return self
        
    def build(self):
        return LogisticRegression(
            bias=self.bias,
            loss=self.loss,
            optimizer=self.optimizer,
            regularizer=self.regularizer,
            weight_matrix=self.weight_matrix
        )