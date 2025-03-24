import sys
from typing import Literal, Optional, Union

import numpy as np

from path_handler import PathManager

path_manager = PathManager()
sys.path.append(str(path_manager.get_base_directory()))

from src.linear.models.svm.base import SVM
from src.linear.components.loss import LossFunction
from src.linear.components.optimizer import Optimizer
from src.linear.components.regularizer import Regularizer
from src.linear.components.factory import ComponentFactory


class SVMBuilder:
    def __init__(self):
        self.bias = np.array([])
        self.loss: Optional[LossFunction] = None
        self.weight_matrix = np.array([])
        self.optimizer: Optional[Optimizer] = None
        self.regularizer: Optional[Regularizer] = None
        
    def compile(
        self, 
        optimizer: Union[Literal["adam", "adagrad", "rmsprop", "sgd", "newton_method"], Optimizer], 
        loss: Union[Literal["hinge"], LossFunction], 
        regularizer: Optional[Union[Literal["l1", "l2", "l1l2"], Regularizer]] = None
    ) -> 'SVMBuilder':

        self.loss = ComponentFactory.create_loss(loss)
        self.optimizer = ComponentFactory.create_optimizer(optimizer)
        self.regularizer = ComponentFactory.create_regularizer(regularizer)
        
        return self
        
    def build(self):
        if self.loss is None or self.metric is None or self.search_algorithm is None:
            raise ValueError("The model is not compiled yet. Please call the compile method before build.")
        
        return SVM(
            bias=self.bias,
            loss=self.loss,
            optimizer=self.optimizer,
            regularizer=self.regularizer,
            weight_matrix=self.weight_matrix
        )