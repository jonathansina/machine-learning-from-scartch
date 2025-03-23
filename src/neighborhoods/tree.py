from typing import Optional

import numpy as np


class KDNode:
    def __init__(
        self, 
        value: np.ndarray, 
        label: np.ndarray, 
        depth: int, 
        rchild: Optional["KDNode"] = None,
        lchild: Optional["KDNode"] = None
    ):
        self.value = value
        self.label = label
        self.depth = depth
        self.rchild = rchild
        self.lchild = lchild


class KDTree:
    def __init__(self, values: np.ndarray, labels: np.ndarray):
        self.values = values
        self.labels = labels
        self.dimensions = self.values.shape[1]
        
        data = np.column_stack((values, labels))
        self.root = self._build_tree(data, 0)

    def _build_tree(self, data, depth) -> Optional[KDNode]:
        if len(data) == 0:
            return None

        current_dimension = depth % self.dimensions
        data = data[data[:, current_dimension].argsort()]
        median_index = len(data) // 2

        node = KDNode(
            value=data[median_index][:-1],
            label=data[median_index][-1],
            depth=depth
        )

        node.lchild = self._build_tree(
            data=data[:median_index],
            depth=depth + 1
        )
        node.rchild = self._build_tree(
            data=data[median_index + 1:],
            depth=depth + 1
        )

        return node
