import sys
from typing import Literal

from path_handler import PathManager

path_manager = PathManager()
sys.path.append(str(path_manager.get_base_directory()))

from src.neighborhood.base import NearestNeighbor
from src.neighborhood.predictor import RegressorPredictor, ClassifierPredictor


class NearestNeighborFactory:
    @staticmethod
    def create(strategy_type: Literal["classifier", "regressor"]) -> NearestNeighbor:
        if strategy_type == "classifier":
            return NearestNeighbor(
                predictor=ClassifierPredictor()
            )
        
        elif strategy_type == "regressor":
            return NearestNeighbor(
                predictor=RegressorPredictor()
            )
        
        else:
            raise ValueError(f"Unknown strategy type: {strategy_type}")
