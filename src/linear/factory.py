import sys
from typing import Literal

from path_handler import PathManager

path_manager = PathManager()
sys.path.append(str(path_manager.get_base_directory()))

from src.linear.models.svm.builder import SVMBuilder
from src.linear.models.linear.builder import LinearRegressionBuilder
from src.linear.models.logistic.builder import LogisticRegressionlBuilder


class LinearModelFactory:
    @staticmethod
    def create(model_type: Literal["linear_regression", "logistic_regression", "svm"]):
        if model_type == "linear_regression":
            return LinearRegressionBuilder()

        elif model_type == "logistic_regression":
            return LogisticRegressionlBuilder()

        elif model_type == "svm":
            return SVMBuilder()

        else:
            raise ValueError(f"Unknown model type: {model_type}")