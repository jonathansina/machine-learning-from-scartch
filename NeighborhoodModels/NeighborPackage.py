import numpy as np
import math


class Euclidean:
    def __init__(self):
        self.name = "euclidean_distance"

    def __call__(self, x: np.ndarray, y: np.ndarray) -> float:
        dist = [(a - b) ** 2 for a, b in zip(x, y)]
        dist = math.sqrt(sum(dist))
        return dist


class Manhattan:
    def __init__(self):
        self.name = "manhattan_distance"

    def __call__(self, x: np.ndarray, y: np.ndarray) -> float:
        dist = [abs(a - b) for a, b in zip(x, y)]
        return sum(dist)


class Chebyshev:
    def __init__(self):
        self.name = "chebyshev_distance"

    def __call__(self, x: np.ndarray, y: np.ndarray) -> float:
        dist = [abs(a - b) for a, b in zip(x, y)]
        return max(dist)


class Minkowski:
    def __init__(self, p: int = 2):
        self.name = "minkowski_distance"
        self.p = p

    def __call__(self, x, y) -> float:
        return sum([(a - b) ** self.p for a, b in zip(x, y)]) ** 1 / self.p


class Hamming:
    def __init__(self):
        self.name = "hamming_distance"

    def __call__(self, x: np.ndarray, y: np.ndarray) -> float:
        if len(x) != len(y):
            raise ValueError("Two vectors must have the same length!")
        distance = 0
        for a, b in zip(x, y):
            if a != b:
                distance += 1
        return distance / len(x)


class Cosine:
    def __init__(self):
        self.name = "cosine_distance"

    def __call__(self, x: np.ndarray, y: np.ndarray) -> float:
        numerator = np.dot(x, y)
        denominator = np.linalg.norm(x) * np.linalg.norm(y)
        return 1 - numerator / denominator


class Jaccard:
    def __init__(self):
        self.name = "jaccard_distance"

    def __call__(self, x: np.ndarray, y: np.ndarray) -> float:
        intersection = len(set(x).intersection(set(y)))
        union = len(set(x).union(set(y)))
        return 1 - intersection / union


class KDNode:
    """
    A node in a KD-Tree.

    Parameters
    ----------
    value : np.ndarray
        The value of the node.
    label : np.ndarray
        The label of the node.
    depth : int
        The depth of the node in the tree.
    rchild : classmethod | None
        The right child of the node.
    lchild : classmethod | None
        The left child of the node.
    """

    def __init__(self, value: np.ndarray, label: np.ndarray, depth: int, rchild: classmethod | None,
                 lchild: classmethod | None):
        self.value = value
        self.label = label
        self.depth = depth
        self.rchild = rchild
        self.lchild = lchild


class KDTree:
    """
    A K-D tree data structure for fast nearest neighbor search.

    Parameters
    ----------
    values : np.ndarray
        The array of values to be stored in the tree.
    labels : np.ndarray
        The array of labels to be stored in the tree.
    dimensions : int
        The number of dimensions of the data.

    Attributes
    ----------
    values : np.ndarray
        The array of values stored in the tree.
    labels : np.ndarray
        The array of labels stored in the tree.
    dimensions : int
        The number of dimensions of the data.
    root : KDNode | None
        The root node of the tree.

    """

    def __init__(self, values: np.ndarray, labels: np.ndarray):
        self.values = values
        self.labels = labels
        self.dimensions = self.values.shape[1]
        data = np.column_stack((values, labels))
        self.root = self.build_tree(data, 0)

    def build_tree(self, data, depth) -> KDNode | None:
        """
        Build a K-D tree from the given data.

        Parameters
        ----------
        data : np.ndarray
            The array of data points.
        depth : int
            The depth of the current node in the tree.

        Returns
        -------
        KDNode | None
            The root node of the tree.

        """
        if len(data) == 0:
            return None
        current_dimension = depth % self.dimensions
        data = data[data[:, current_dimension].argsort()]
        median_index = len(data) // 2
        node = KDNode(
            value=data[median_index][:-1],
            label=data[median_index][-1],
            depth=depth,
            rchild=None,
            lchild=None
        )
        node.lchild = self.build_tree(
            data=data[:median_index],
            depth=depth + 1
        )
        node.rchild = self.build_tree(
            data=data[median_index + 1:],
            depth=depth + 1
        )
        return node


class NearestNeighbor:
    """
    A class for finding the nearest neighbors in a dataset.

    Parameters
    ----------
    k : int
        The number of nearest neighbors to find.
    metrics : str | DistanceMetric
        The distance metric to use for finding the nearest neighbors.
        Can be one of the following strings: "euclidean", "manhattan", "chebyshev", "minkowski", "cosine", "jaccard", or "hamming",
        or a custom DistanceMetric object.
    algorithm : str
        The algorithm to use for finding the nearest neighbors.
        Can be either "brute-force" or "kd-tree".

    Attributes
    ----------
    knn_set : list
        A list of tuples containing the nearest neighbors and their distances.
    tree : KDTree | None
        The KDTree object used for finding the nearest neighbors if the "kd-tree" algorithm is selected.
    algorithm : str
        The algorithm used for finding the nearest neighbors.
    x_train : np.ndarray | None
        The training data points if the "brute-force" algorithm is selected.
    y_train : np.ndarray | None
        The training labels if the "brute-force" algorithm is selected.
    metrics : DistanceMetric
        The distance metric used for finding the nearest neighbors.

    Methods
    -------
    compile(k, metrics, algorithm)
        Compiles the model with the specified parameters.
    find_knn(x)
        Finds the k nearest neighbors of the given data point x.
    train(x_train, y_train)
        Trains the model on the given training data.
    predict(x)
        Predicts the label of the given data point x.
    brute_force(x)
        Uses the brute-force algorithm to find the nearest neighbors.
    kd_tree(x, root)
        Uses the KD-Tree algorithm to find the nearest neighbors.
    """

    def __init__(self, type: str = "classifier"):
        """
        Initialize the NearestNeighbor class.

        Parameters
        ----------
        type : str, optional
            The type of model, either "classifier" or "regressor", by default "classifier".
        """
        if type == "classifier" or type == "regressor":
            self.type = type
        else:
            raise ValueError("type must be either 'classifier' or 'regressor'")
        self.knn_set = []
        self.tree = None
        self.algorithm = None
        self.x_train = None
        self.y_train = None
        self.k = None
        self.metrics = None

    def compile(self, k: int, metrics: str = "euclidean", algorithm: str = "brute-force"):
        """
        Compile the model with the specified parameters.

        Parameters
        ----------
        k : int
            The number of nearest neighbors to find.
        metrics : str | DistanceMetric
            The distance metric to use for finding the nearest neighbors.
            Can be one of the following strings: "euclidean", "manhattan", "chebyshev", "minkowski", "cosine", "jaccard", or "hamming",
            or a custom DistanceMetric object.
        algorithm : str
            The algorithm to use for finding the nearest neighbors.
            Can be either "brute-force" or "kd-tree".
        """
        if metrics == "euclidean":
            self.metrics = Euclidean()
        elif metrics == "manhattan":
            self.metrics = Manhattan()
        elif metrics == "chebyshev":
            self.metrics = Chebyshev()
        elif metrics == "minkowski":
            self.metrics = Minkowski()
        elif metrics == "cosine":
            self.metrics = Cosine()
        elif metrics == "jaccard":
            self.metrics = Jaccard()
        elif metrics == "hamming":
            self.metrics = Hamming()
        elif isinstance(metrics, (Euclidean, Minkowski, Manhattan, Cosine, Chebyshev, Hamming, Jaccard)):
            self.metrics = metrics
        else:
            raise ValueError("Metrics invalid!")

        if algorithm == "brute-force" or algorithm == "kd-tree":
            self.algorithm = algorithm

        self.k = k

    def find_knn(self, x: np.ndarray) -> np.ndarray:
        """
        Find the k nearest neighbors of the given data point x.

        Parameters
        ----------
        x : np.ndarray
            The data point for which to find the nearest neighbors.

        Returns
        -------
        np.ndarray
            The indices of the k nearest neighbors of x.
        """
        distances = []
        for i in range(self.x_train.shape[0]):
            distances.append(self.metrics(self.x_train[i], x))

        return np.argsort(distances)[:self.k]

    def train(self, x_train: np.ndarray, y_train: np.ndarray):
        """
        Train the model on the given training data.

        Parameters
        ----------
        x_train : np.ndarray
            The training data points.
        y_train : np.ndarray
            The training labels.
        """
        if self.algorithm == "kd-tree":
            kdtree = KDTree(
                values=x_train,
                labels=y_train
            )
            self.tree = kdtree
            print("KD-Tree Constructed Successfully!")

        elif self.algorithm == "brute-force":
            self.x_train = x_train
            self.y_train = y_train

    def predict(self, x: np.ndarray) -> np.ndarray | int:
        """
        Predict the label of the given data point x.

        Parameters
        ----------
        x : np.ndarray
            The data point for which to predict the label.

        Returns
        -------
        np.ndarray | int
            The predicted label of x, or an array of predicted labels if x is an array.
        """
        if x.T.shape != x.shape and x.shape[0] != 1:
            return np.array([self.predict(x[i]) for i in range(x.shape[0])])

        if self.algorithm == "brute-force":
            return self.brute_force(x)

        elif self.algorithm == "kd-tree":
            self.knn_set = []
            labels = []
            self.kd_tree(x, self.tree.root)
            for node, _ in self.knn_set:
                labels.append(node.label)

            if self.type == 'classifier':
                predicted_label = np.argmax(np.bincount(labels))
            else:
                predicted_label = np.mean(labels)
            return predicted_label

    def brute_force(self, x: np.ndarray) -> int:
        """
        Use the brute-force algorithm to find the nearest neighbors.

        Parameters
        ----------
        x : np.ndarray
            The data point for which to find the nearest neighbors.

        Returns
        -------
        int
            The index of the nearest neighbor of x.
        """
        distances = self.find_knn(x)
        labels = []
        for i in distances:
            labels.append(self.y_train[i])

        if self.type == 'classifier':
            predicted_label = np.argmax(np.bincount(labels))
        else:
            predicted_label = np.mean(labels)
        return predicted_label

    def kd_tree(self, x: np.ndarray, root: classmethod | KDNode):
        """
        Use the KD-Tree algorithm to find the nearest neighbors.

        Parameters
        ----------
        x : np.ndarray
            The data point for which to find the nearest neighbors.
        root : classmethod | KDNode
            The root node of the KD-Tree.
        """
        if root is None:
            return
        label = root.label
        current_value = root.value
        distance = self.metrics(x, current_value)

        duplicate = [self.metrics(current_value, item[0].value) < 1e-4 and
                     abs(label - item[0].label) for item in self.knn_set]

        if not np.array(duplicate, bool).any():
            if len(self.knn_set) < self.k:
                self.knn_set.append((root, distance))
            elif distance < self.knn_set[0][1]:
                self.knn_set[0] = (root, distance)

        self.knn_set = sorted(self.knn_set, key=lambda x: -x[1])
        current_dimension = root.depth % self.tree.dimensions
        if abs(x[current_dimension] - current_value[current_dimension]) < self.knn_set[0][1]:
            self.kd_tree(x, root.lchild)
            self.kd_tree(x, root.rchild)
        elif x[current_dimension] < current_value[current_dimension]:
            self.kd_tree(x, root.lchild)
        else:
            self.kd_tree(x, root.rchild)
