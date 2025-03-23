from typing import Optional, Literal

import numpy as np


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