import sys
from typing import Literal

from path_handler import PathManager

path_manager = PathManager()
sys.path.append(str(path_manager.get_base_directory()))

from src.neighborhood.builder import NearestNeighborBuilder
from src.neighborhood.components.predictor import RegressorPredictor, ClassifierPredictor


class NearestNeighborFactory:
    @staticmethod
    def create(strategy_type: Literal["classifier", "regressor"]) -> NearestNeighborBuilder:
        if strategy_type == "classifier":
            return NearestNeighborBuilder(
                predictor=ClassifierPredictor()
            )
        
        elif strategy_type == "regressor":
            return NearestNeighborBuilder(
                predictor=RegressorPredictor()
            )
        
        else:
            raise ValueError(f"Unknown strategy type: {strategy_type}")
