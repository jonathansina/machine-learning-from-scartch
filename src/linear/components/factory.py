import sys
from typing import Union, Optional

from path_handler import PathManager

path_manager = PathManager()
sys.path.append(str(path_manager.get_base_directory()))

from src.linear.components.optimizer import Optimizer, AdaGrad, RMSProp, ADAM, SGD, NewtonMethod
from src.linear.components.regularizer import Regularizer, L1Regularizer, L2Regularizer, ElasticNetRegularizer
from src.linear.components.loss import Loss, MeanSquaredError, MeanAbsoluteError, BinaryCrossEntropy, Hinge, LogLoss, Huber


class ComponentFactory:
    REGULARIZER_MAP = {
        'l1': L1Regularizer(),
        'l2': L2Regularizer(),
        'l1l2': ElasticNetRegularizer()
    }
    OPTIMIZER_MAP = {
        'sgd': SGD(),
        'adam': ADAM(),
        'adagrad': AdaGrad(),
        'rmsprop': RMSProp(),
        'gradient-descent': SGD(),
        'newton-method': NewtonMethod()
    }
    LOSS_MAP = {
        'huber': Huber(),
        'hinge': Hinge(),
        'log_loss': LogLoss(),
        'mse': MeanSquaredError(),
        'mae': MeanAbsoluteError(),
        'binary_crossentropy': BinaryCrossEntropy()
    }

    @classmethod
    def create_optimizer(cls, optimizer: Union[str, Optimizer]) -> Optimizer:
        if isinstance(optimizer, Optimizer):
            return optimizer

        if optimizer in cls.OPTIMIZER_MAP:
            return cls.OPTIMIZER_MAP[optimizer]
        
        raise ValueError(f'Invalid optimizer: {optimizer}')
    
    @classmethod
    def create_regularizer(cls, regularizer: Union[str, Regularizer, None]) -> Optional[Regularizer]:
        if regularizer is None:
            return None
            
        if isinstance(regularizer, Regularizer):
            return regularizer
        
        if regularizer in cls.REGULARIZER_MAP:
            return cls.REGULARIZER_MAP[regularizer]
            
        raise ValueError(f'Invalid regularizer: {regularizer}')
    
    @classmethod
    def create_loss(cls, loss: Union[str, Loss, None]) -> Loss:
        if loss is None:
            return None

        if isinstance(loss, Loss):
            return loss

        if loss in cls.LOSS_MAP:
            return cls.LOSS_MAP[loss]
        
        raise ValueError(f'Invalid loss function: {loss}')