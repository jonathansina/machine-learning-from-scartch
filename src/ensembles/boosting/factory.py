import sys
from typing import Literal, Union

from path_handler import PathManager

path_manager = PathManager()
sys.path.append(str(path_manager.get_base_directory()))

from src.ensembles.boosting.models.gradient.builder import BoostingRegressorBuilder
from src.ensembles.boosting.models.adaboost.builder import BoostingClassifierBuilder


class BoostingFactory:
    @staticmethod
    def create(strategy_type: Literal["regressor", "classifier"]) -> Union[BoostingRegressorBuilder, BoostingClassifierBuilder]:
        if strategy_type == "regressor":
            return BoostingRegressorBuilder()
        
        elif strategy_type == "classifier":
            return BoostingClassifierBuilder()
        
        else:
            raise ValueError(f"Invalid strategy type: {strategy_type}")