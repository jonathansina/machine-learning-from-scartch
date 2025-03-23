from typing import Tuple

import numpy as np


class TreeUtils:
    @staticmethod
    def make_split(x: np.ndarray, threshold: float) -> Tuple[np.ndarray, np.ndarray]:
        left_indices = np.argwhere(x <= threshold).flatten()
        right_indices = np.argwhere(x > threshold).flatten()
        return left_indices, right_indices
