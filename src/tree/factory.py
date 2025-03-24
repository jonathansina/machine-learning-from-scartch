import sys
from typing import Literal

from path_handler import PathManager

path_manager = PathManager()
sys.path.append(str(path_manager.get_base_directory()))

from src.tree.builder import IdentificationTreeBuilder
from src.tree.components.strategy import ClassificationTreeBuilder, RegressionTreeBuilder


class IdentificationTreeFactory:
    @staticmethod
    def create(strategy_type: Literal["classifier", "regressor"], ) -> IdentificationTreeBuilder:
        if strategy_type == "classifier":
            builder_strategy = ClassificationTreeBuilder()
        
        elif strategy_type == "regressor":
            builder_strategy = RegressionTreeBuilder()
        
        else:
            raise ValueError(f"Unknown strategy type: {strategy_type}")
        
        return IdentificationTreeBuilder(builder_strategy=builder_strategy)