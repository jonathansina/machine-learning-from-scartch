import sys
from typing import Literal

from sklearn.metrics import r2_score, accuracy_score

from path_handler import PathManager

path_manager = PathManager()
sys.path.append(str(path_manager.get_base_directory()))

from src.ensembles.bagging.builder import BaggingBuilder
from src.ensembles.bagging.components.aggregator import ClassifierAggregator, RegressorAggregator


class BaggingFactory:
    @staticmethod
    def create(strategy_type: Literal["classifier", "regressor"]) -> BaggingBuilder:
        if strategy_type == "classifier":
            return BaggingBuilder(
                aggregator=ClassifierAggregator(), 
                error_metric=accuracy_score
            )
        
        elif strategy_type == "regressor":
            return BaggingBuilder(
                aggregator=RegressorAggregator(), 
                error_metric=r2_score
            )
        
        else:
            raise ValueError(f"Unknown strategy type: {strategy_type}")