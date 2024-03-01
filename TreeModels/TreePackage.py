import numpy as np
from sklearn.metrics import r2_score, accuracy_score
from LinearPackage import MeanAbsoluteError, Huber, MeanSquaredError


class Gini(object):
    """
    Calculates the Gini impurity index of a set of probabilities.

    Parameters
    ----------
    set : np.ndarray
        The set of probabilities.

    Returns
    -------
    float
        The Gini impurity index of the set.

    """

    def __init__(self):
        self.name = "gini_index"

    def __call__(self, set: np.ndarray) -> float:
        set = np.array(set, dtype=np.int64)
        prob = np.bincount(set) / len(set)
        return sum(prob * (1 - prob))


class Entropy(object):
    """
    Calculates the Shannon entropy of a set of probabilities.

    Parameters
    ----------
    set : np.ndarray
        The set of probabilities.

    Returns
    -------
    float
        The Shannon entropy of the set.

    """
    def __init__(self):
        self.name = "entropy"

    def __call__(self, set: np.ndarray):
        set = np.array(set, dtype=np.int64)
        prob = np.bincount(set) / len(set)
        return -sum(p * np.log2(p) for p in prob if p > 0)


class Node(object):
    """
    A node in a tree data structure.

    Parameters
    ----------
    dimension : int
        The dimension used for the split.
    threshold : float
        The split threshold.
    depth : int
        The depth of the node.
    lchild : Node
        The left child node.
    rchild : Node
        The right child node.
    value : int
        The leaf value.

    Attributes
    ----------
    dimension : int
        The dimension used for the split.
    threshold : float
        The split threshold.
    depth : int
        The depth of the node.
    lchild : Node
        The left child node.
    rchild : Node
        The right child node.
    value : int
        The leaf value.
    """

    def __init__(self, dimension: int = None, threshold: float = None, depth: int = None, lchild=None,
                 rchild=None, value: int = None):
        self.dimension = dimension
        self.threshold = threshold
        self.depth = depth
        self.lchild = lchild
        self.rchild = rchild
        self.value = value

    def is_leaf(self) -> bool:
        """
        Check if the node is a leaf.

        Returns
        -------
        bool
            True if the node is a leaf, False otherwise.
        """
        return self.value is not None


class Tree(object):
    """
    A class for creating decision trees.

    Parameters
    ----------
    values : np.ndarray
        The input data values.
    labels : np.ndarray
        The input data labels.
    impurity_function : function
        The impurity function used to calculate node impurities.
    max_depth : int
        The maximum tree depth.
    number_of_dimensions : int, optional
        The number of dimensions to use for splitting, by default None
    type : str, optional
        The type of tree (classifier or regressor), by default "classifier"

    Attributes
    ----------
    values : np.ndarray
        The input data values.
    labels : np.ndarray
        The input data labels.
    impurity_function : function
        The impurity function used to calculate node impurities.
    max_depth : int
        The maximum tree depth.
    dimensions : list
        The list of dimensions used for splitting.
    depth : int
        The current tree depth.
    root : Node
        The root node of the tree.

    """

    def __init__(self, values: np.ndarray, labels: np.ndarray, impurity_function, max_depth: int,
                 number_of_dimensions: int = None, type: str = "classifier"):
        self.values = values
        self.labels = labels
        self.dimensions = np.random.choice(self.values.shape[1], number_of_dimensions, replace=False)
        data = np.column_stack((values, labels))
        self.impurity_function = impurity_function
        self.max_depth = max_depth
        self.depth = 0
        self.type = type
        self.root = self.build_tree(data, 0)

    def build_tree(self, data: np.ndarray, depth: int) -> Node:
        """
        Build the decision tree.

        Parameters
        ----------
        data : np.ndarray
            The input data.
        depth : int
            The current tree depth.

        Returns
        -------
        Node
            The root node of the tree.

        """
        repeated_data = len(np.unique(data[:, -1])) == 2 and (data[0, :-1] == data[1, :-1]).all() and (data[0, -1] != data[1, -1]).all()
        if len(np.unique(data[:, -1])) == 1 or repeated_data or depth >= self.max_depth:
            if self.type == "classifier":
                leaf_value = self.most_common_label(data[:, -1])
            else:
                leaf_value = np.mean(data[:, -1])

            if self.depth < depth:
                self.depth = depth
            return Node(value=leaf_value, depth=depth)

        best_dim, best_threshold = self.best_split(data)
        left_indexes, right_indexes = self.make_split(data[:, best_dim], best_threshold)

        node = Node(
            dimension=best_dim,
            threshold=best_threshold,
            depth=depth,
            rchild=None,
            lchild=None
        )
        node.lchild = self.build_tree(
            data=data[left_indexes, :],
            depth=depth + 1
        )
        node.rchild = self.build_tree(
            data=data[right_indexes, :],
            depth=depth + 1
        )
        return node

    def best_split(self, data: np.ndarray) -> tuple[int, float]:
        """
        Find the best split for the data.

        Parameters
        ----------
        data : np.ndarray
            The input data.

        Returns
        -------
        tuple[int, float]
            The best split dimension and threshold.

        """
        best_gain = -(10**13)
        split_dimension, split_threshold = None, None
        for i in self.dimensions:
            thresholds = np.unique(data[:, i])
            for j in thresholds:
                gain = self.information_gain(data, j, i)
                if gain > best_gain:
                    best_gain = gain
                    split_dimension = i
                    split_threshold = j

        return split_dimension, split_threshold

    def information_gain(self, data: np.ndarray, threshold: float, dimension: int) -> float:
        """
        Calculate the information gain for a split.

        Parameters
        ----------
        data : np.ndarray
            The input data.
        threshold : float
            The split threshold.
        dimension : int
            The split dimension.

        Returns
        -------
        float
            The information gain.

        """
        if self.type == "classifier":
            parent_gain = self.impurity_function(data[:, -1])
            left_indexes, right_indexes = self.make_split(data[:, dimension], threshold)

            len_left_indexes = len(left_indexes)
            len_right_indexes = len(right_indexes)
            len_indexes = len(data[:, -1])

            left_gain = self.impurity_function(data[left_indexes, -1])
            right_gain = self.impurity_function(data[right_indexes, -1])
            child_gain = (len_left_indexes / len_indexes) * left_gain + (
                    len_right_indexes / len_indexes) * right_gain
            return parent_gain - child_gain

        else:
            parent_gain = self.impurity_function(data[:, -1], np.mean(data[:, -1]))
            left_indexes, right_indexes = self.make_split(data[:, dimension], threshold)

            target_value_left = np.mean(data[left_indexes, -1])
            target_value_right = np.mean(data[right_indexes, -1])

            len_left_indexes = len(left_indexes)
            len_right_indexes = len(right_indexes)
            len_indexes = len(data[:, -1])

            left_gain = self.impurity_function(data[left_indexes, -1], target_value_left)
            right_gain = self.impurity_function(data[right_indexes, -1], target_value_right)

            child_gain = (len_left_indexes / len_indexes) * left_gain + (
                        len_right_indexes / len_indexes) * right_gain

            return parent_gain - child_gain

    @staticmethod
    def make_split(x: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Make the split based on the split threshold.

        Parameters
        ----------
        x : np.ndarray
            The input data values.
        threshold : float
            The split threshold.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            The left and right indexes of the split.

        """
        left_indexes = np.argwhere(x <= threshold).flatten()
        right_indexes = np.argwhere(x > threshold).flatten()
        return left_indexes, right_indexes

    @staticmethod
    def most_common_label(y: np.ndarray) -> int:
        """
        Find the most common label in the data.

        Parameters
        ----------
        y : np.ndarray
            The input data labels.

        Returns
        -------
        int
            The most common label.

        """
        y = np.array(y, dtype=np.int64)
        most_common = np.argmax(np.bincount(y))
        return most_common


class IdentificationTree(object):
    """
    A class for creating decision trees.

    Parameters
    ----------
    type : str, optional
        The type of tree (classifier or regressor), by default "classifier"

    Attributes
    ----------
    type : str
        The type of tree (classifier or regressor)
    max_features : int
        The maximum number of features to consider for splitting at each node
    tree : Tree
        The decision tree
    impurity_function : function
        The impurity function used to calculate node impurities
    max_depth : int
        The maximum tree depth
    """

    def __init__(self, type: str = "classifier"):
        self.type = type
        self.max_features = None
        self.tree = None
        self.impurity_function = None
        self.max_depth = None

    def compile(self, max_depth: int, impurity_function: str | None, max_features: int = None):
        """
        Compile the parameters for the tree.

        Parameters
        ----------
        max_depth : int
            The maximum tree depth.
        impurity_function : str or None
            The impurity function to use.
        max_features : int, optional
            The maximum number of features to consider for splitting, by default -1
        """
        if self.type == "classifier":
            if impurity_function == "entropy":
                self.impurity_function = Entropy()
            elif impurity_function == "gini":
                self.impurity_function = Gini()
            else:
                raise ValueError("Invalid impurity function!")
        elif self.type == "regressor":
            if impurity_function == "mse":
                self.impurity_function = MeanSquaredError()
            elif impurity_function == "mae":
                self.impurity_function = MeanAbsoluteError()
            elif impurity_function == "huber":
                self.impurity_function = Huber()
            elif isinstance(impurity_function, (MeanSquaredError, MeanAbsoluteError, Huber)):
                self.impurity_function = impurity_function
            else:
                raise ValueError("Invalid impurity function!")
        self.max_depth = max_depth
        self.max_features = max_features

    def train(self, x_train: np.ndarray, y_train: np.ndarray):
        """
        Train the tree on the given data.

        Parameters
        ----------
        x_train : np.ndarray
            The input data values.
        y_train : np.ndarray
            The input data labels.
        """
        if self.max_features is None:
            self.max_features = x_train.shape[1]
        elif self.max_features == "sqrt":
            self.max_features = int(np.sqrt(x_train.shape[1])) + 1
        elif self.max_features == "log":
            self.max_features = int(np.log(x_train.shape[1])) + 1

        self.tree = Tree(
            values=x_train,
            labels=y_train,
            impurity_function=self.impurity_function,
            max_depth=self.max_depth,
            number_of_dimensions=self.max_features,
            type=self.type
        )

    def predict(self, x: np.ndarray) -> np.ndarray | int:
        """
        Make a prediction based on the given data.

        Parameters
        ----------
        x : np.ndarray
            The input data.

        Returns
        -------
        np.ndarray or int
            The predicted value or values.
        """
        if x.T.shape != x.shape and x.shape[0] != 1:
            return np.array([self.predict(x[i]) for i in range(x.shape[0])])

        return self.traverse(x, self.tree.root)

    def traverse(self, x: np.ndarray, node: Node | classmethod) -> int:
        """
        Traverse the tree to make a prediction.

        Parameters
        ----------
        x : np.ndarray
            The input data.
        node : Node or classmethod
            The current node.

        Returns
        -------
        int
            The predicted value.
        """
        if node.is_leaf():
            return node.value

        if x[node.dimension] <= node.threshold:
            return self.traverse(x, node.lchild)
        else:
            return self.traverse(x, node.rchild)
