import sys
from typing import Union, Literal

from path_handler import PathManager

path_manager = PathManager()
sys.path.append(str(path_manager.get_base_directory()))

from src.ensembles.bagging.base import BaggingClassifier, BaggingRegressor


class BaggingFactory:
    @staticmethod
    def create(strategy_type: Literal["classifier", "regressor"]) -> Union[BaggingClassifier, BaggingRegressor]:
        if strategy_type == "classifier":
            return BaggingClassifier()
        
        elif strategy_type == "regressor":
            return BaggingRegressor()
        
        else:
            raise ValueError(f"Unknown strategy type: {strategy_type}")