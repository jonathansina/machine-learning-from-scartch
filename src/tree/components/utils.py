from typing import Tuple, Optional, Literal

import numpy as np


class TreeUtils:
    @staticmethod
    def make_split(x: np.ndarray, threshold: float) -> Tuple[np.ndarray, np.ndarray]:
        left_indices = np.argwhere(x <= threshold).flatten()
        right_indices = np.argwhere(x > threshold).flatten()
        return left_indices, right_indices


class FeatureUtils:
    @staticmethod
    def get_feature_count(max_features: Optional[Literal["sqrt", "log"]], n_features: int):
        if max_features is None:
            return n_features

        elif max_features == "sqrt":
            return int(np.sqrt(n_features)) + 1

        elif max_features == "log":
            return int(np.log(n_features)) + 1

        return max_features