import sys

from path_handler import PathManager

path_manager = PathManager()
sys.path.append(str(path_manager.get_base_directory()))

from src.tunning.asquisition.base import AcquisitionFunction, UCB, POI, EI


class AcquisitionFunctionFactory:
    @staticmethod
    def create(acquisition_type: str, **kwargs) -> AcquisitionFunction:
        if acquisition_type == "ucb":
            beta = kwargs.get('beta', 2)
            return UCB(beta=beta)

        elif acquisition_type == "poi":
            return POI()
        
        elif acquisition_type == "ei":
            return EI()

        elif isinstance(acquisition_type, AcquisitionFunction):
            return acquisition_type

        raise ValueError(f"Invalid acquisition function: {acquisition_type}")
