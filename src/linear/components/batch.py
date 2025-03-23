from typing import List, Tuple

import numpy as np


class MiniBatchGenerator:
    @staticmethod
    def create_mini_batches(x: np.ndarray, y: np.ndarray, batch_size: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        mini_batches = []
        data = np.hstack((x, y))
        np.random.shuffle(data)
        n_minibatches = data.shape[0] // batch_size
        
        for i in range(n_minibatches):
            mini_batch = data[i * batch_size:(i + 1) * batch_size, :]
            x_mini = mini_batch[:, :-1]
            y_mini = mini_batch[:, -1].reshape((-1, 1))
            mini_batches.append((x_mini, y_mini))
            
        if data.shape[0] % batch_size != 0:
            mini_batch = data[n_minibatches * batch_size:data.shape[0]]
            x_mini = mini_batch[:, :-1]
            y_mini = mini_batch[:, -1].reshape((-1, 1))
            mini_batches.append((x_mini, y_mini))
            
        return mini_batches