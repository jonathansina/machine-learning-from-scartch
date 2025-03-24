import sys
from abc import ABC, abstractmethod
from typing import Optional, Union, Literal, List

import numpy as np
import matplotlib.pyplot as plt

from path_handler import PathManager

path_manager = PathManager()
sys.path.append(str(path_manager.get_base_directory()))

from src.linear.components.loss import LossFunction
from src.linear.components.optimizer import Optimizer
from src.linear.components.regularizer import Regularizer
from src.linear.components.batch import MiniBatchGenerator
from src.linear.components.factory import ComponentFactory
from src.linear.trainer import MiniBatchTrainingStrategy, SGDTrainingStrategy


class BaseLinearModel(ABC):
    def __init__(self):
        self.epochs = 0
        self.cost: List[float] = []
        self.bias = np.array([])
        self.loss: Optional[LossFunction] = None
        self.weight_matrix = np.array([])
        self.optimizer: Optional[Optimizer] = None
        self.batch_generator = MiniBatchGenerator()
        self.regularizer: Optional[Regularizer] = None
        
    def compile(
        self, 
        optimizer: Union[Literal["adam", "adagrad", "rmsprop", "sgd", "newton_method"], Optimizer], 
        loss: Union[Literal["mse", "mae", "huber", "binary_crossentropy", "log_loss", "hinge"], LossFunction], 
        regularizer: Optional[Union[Literal["l1", "l2", "l1l2"], Regularizer]] = None
    ):
        self.loss = ComponentFactory.create_loss(loss)
        self.optimizer = ComponentFactory.create_optimizer(optimizer)
        self.regularizer = ComponentFactory.create_regularizer(regularizer)
        
        self._validate_loss_function(self.loss)

    def fit(self, x: np.ndarray, y: np.ndarray, epochs: int, batch_size: int = 1, verbose: int = 2):
        if self.loss is None:
            raise ValueError("The model is not compiled yet. Please call the compile method before fit.")

        self.epochs += epochs

        if y.ndim == 1:
            y = y.reshape(y.shape[0], 1)

        self._initialize_parameters(x, y)
        batch_size = self._determine_batch_size(x, batch_size)
        
        training_strategy = self._create_training_strategy(batch_size)
        mini_batches = self.batch_generator.create_mini_batches(x, y, batch_size)
        
        for epoch in range(1, epochs + 1):
            cost = training_strategy.train_epoch(mini_batches, self)
            self.cost.append(cost)
            
            if verbose >= 2:
                self._print_epoch_info(epoch, epochs, cost)
                
        if verbose >= 1:
            self._plot_loss(epochs)
            
        if verbose >= 0:
            print('Optimization Finished')
    
    def _initialize_parameters(self, x: np.ndarray, y: np.ndarray):
        self.weight_matrix = np.random.uniform(-1, 1, size=(y.shape[1], x.shape[1]))
        self.bias = np.random.uniform(-1, 1, size=(y.shape[1]))
    
    def _determine_batch_size(self, x: np.ndarray, requested_batch_size: int) -> int:
        if self.optimizer.name == "newton_method":
            return x.shape[0]

        return requested_batch_size
    
    def _create_training_strategy(self, batch_size: int):
        if batch_size == 1:
            return SGDTrainingStrategy(self.optimizer, self.loss, self.regularizer)

        return MiniBatchTrainingStrategy(self.optimizer, self.loss, self.regularizer)

    def _plot_loss(self):
        plt.plot(range(self.epochs), self.cost)
        plt.xlabel('Number of Epochs')
        plt.ylabel('Cost')
        plt.title('Cost Function')
        plt.grid(True)
    
    def _print_epoch_info(self, current_epoch: int, total_epochs: int, cost: float):
        print(f"-------------------[EPOCH {current_epoch}/{total_epochs}]---------------------")
        print(f'Error: {cost}')
    
    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def _validate_loss_function(self, loss):
        raise NotImplementedError("Subclasses must implement this method")