import numpy as np
from typing import Optional, Tuple, List


class Bootstrap:
    @staticmethod
    def create_samples(
        x: np.ndarray, 
        y: np.ndarray, 
        num_estimators: int, 
        max_samples: Optional[int] = None
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:

        max_samples = x.shape[0] if max_samples is None else max_samples
        bootstrap_samples_x = []
        bootstrap_samples_y = []
        
        for _ in range(num_estimators):
            samples_indexes = np.random.choice(x.shape[0], size=max_samples, replace=True)
            bootstrap_samples_x.append(x[samples_indexes])
            bootstrap_samples_y.append(y[samples_indexes])
            
        return bootstrap_samples_x, bootstrap_samples_y