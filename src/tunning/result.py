from typing import Optional, List

import numpy as np
import pandas as pd


class OptimizationResult:
    def __init__(self):
        self.data: Optional[pd.DataFrame] = None
    
    def create_dataframe(
        self, 
        indices: List[int], 
        epoch_list: List[int], 
        x_train: np.ndarray, 
        y_train: np.ndarray, 
        size_x: int
    ) -> pd.DataFrame:

        data = np.column_stack((epoch_list, y_train, x_train.reshape(-1, size_x)))
        self.data = pd.DataFrame(data=data, columns=indices)

        return self.data