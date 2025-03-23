from typing import Any, Optional


class Node:
    def __init__(
        self, 
        dimension: Optional[int] = None, 
        threshold: Optional[float] = None, 
        depth: Optional[int] = None, 
        left_child: Optional['Node'] = None,
        right_child: Optional['Node'] = None, 
        value: Optional[Any] = None
    ):
        self.dimension = dimension
        self.threshold = threshold
        self.depth = depth
        self.left_child = left_child
        self.right_child = right_child
        self.value = value

    def is_leaf(self) -> bool:
        return self.value is not None
