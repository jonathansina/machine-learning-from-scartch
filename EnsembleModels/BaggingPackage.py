import numpy as np
from TreePackage import IdentificationTree
import copy


class Bagging(object):
    """
    The Bagging class is used for ensemble learning by combining the predictions of multiple base learners.
    The base learners are trained using a bootstrap sample of the training data, and the predictions are aggregated to form the final prediction.
    The base learners can be of any type, such as decision trees, neural networks, or regression models.
    The Bagging class provides an easy way to improve the performance of any machine learning model without requiring any additional tuning.
    """

    def __init__(self, type: str = "classifier"):
        """
        The __init__ method initializes the Bagging class.
        Args:
            type (str, optional): The type of model to be used. Can be "classifier" or "regressor". Defaults to "classifier".
        """
        if type == "classifier":
            self.type = "classifier"
            self.name = "BaggingClassifier"
        elif type == "regressor":
            self.type = "regressor"
            self.name = "BaggingRegressor"
        else:
            raise ValueError("Type Invalid!")

        self.estimator_stack = None
        self.max_samples = None
        self.max_features = None
        self.number_of_estimators = None
        self.estimator = None

    def compile(self, estimator: classmethod, number_of_estimators: int = 10, max_features: int | str = None,
                max_samples: int = None):
        """
        The compile method sets the parameters for the Bagging class.
        Args:
            estimator (classmethod): The classmethod of the base learner to be used.
            number_of_estimators (int, optional): The number of base learners to be used. Defaults to 10.
            max_features (int or str, optional): The maximum number of features to consider when looking for the best split in each base learner. If "sqrt", then max_features=sqrt(n_features). If "log2", then max_features=log2(n_features). If None, then max_features=n_features. Defaults to None.
            max_samples (int, optional): The maximum number of samples to draw from the training set to train each base learner. If None, then max_samples=n_samples. Defaults to None.
        """
        self.estimator = estimator
        self.number_of_estimators = number_of_estimators
        self.max_features = max_features
        self.max_samples = max_samples
        self.estimator_stack = []

    def train(self, x_train: np.ndarray, y_train: np.ndarray, verbose: int = 2):
        """
        The train method trains the Bagging class using the specified training data.
        Args:
            x_train (np.ndarray): The training data.
            y_train (np.ndarray): The corresponding training labels.
            verbose (int, optional): The level of verbosity. 0 for no output, 1 for progress bar, 2 for one line per iteration. Defaults to 2.
        """
        if self.max_features is None:
            self.max_features = x_train.shape[1]
        elif self.max_features == "sqrt":
            self.max_features = int(np.sqrt(x_train.shape[1]))
        elif self.max_features == "log":
            self.max_features = int(np.log(x_train.shape[1]))

        x_samples, y_samples = self.bootstrap_aggregating(x_train, y_train)
        for i in range(self.number_of_estimators):
            tree = copy.deepcopy(self.estimator)
            tree.train(x_samples[i], y_samples[i])
            self.estimator_stack.append(tree)
            if verbose <= 2:
                print(f"---------------------- Learner {i + 1}/{self.number_of_estimators} Learned ----------------------")
                if verbose <= 1:
                    y_pred = self.estimator_stack[i].predict(x_samples[i])
                    training_error = accuracy_score(y_samples[i], y_pred)
                    print(f"Training Error: {training_error}")

    def predict(self, x: np.ndarray) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """
        The predict method makes predictions using the trained Bagging class.
        Args:
            x (np.ndarray): The data to be predicted.
        Returns:
            np.ndarray or tuple[np.ndarray, np.ndarray]: The predicted values or a tuple of the predicted values and their standard deviation.
        """
        def base_estimator_predict(x):
            predicted_list = []
            for base_model in self.estimator_stack:
                predicted_list.append(base_model.predict(x))
            return predicted_list

        if x.T.shape != x.shape and x.shape[0] != 1:
            return np.array([self.predict(x[i]) for i in range(x.shape[0])])

        predicted_list = base_estimator_predict(x)
        if self.name == "BaggingClassifier":
            y = np.array(predicted_list, dtype=np.int64)
            result = np.argmax(np.bincount(y))
            accuracy = np.bincount(y)[result]/len(y)*100
        else:
            result = np.mean(predicted_list)
            accuracy = np.std(predicted_list)
        return result, accuracy

    def bootstrap_aggregating(self, x: np.ndarray, y:np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """
        The bootstrap_aggregating method generates bootstrap samples for each base learner.
        Args:
            x (np.ndarray): The training data.
            y (np.ndarray): The corresponding training labels.
        Returns:
            tuple[list[np.ndarray], list[np.ndarray]]: A tuple of lists, where each list contains the bootstrap samples for a specific base learner.
        """
        bootstrap_samples_x = []
        bootstrap_samples_y = []
        if self.max_samples is None:
            self.max_samples = x.shape[0]
        for i in range(self.number_of_estimators):
            samples_indexes = np.random.choicse(x.shape[0], size=self.max_samples, replace=True)
            bootstrap_samples_x.append(x[samples_indexes])
            bootstrap_samples_y.append(y[samples_indexes])

        return bootstrap_samples_x, bootstrap_samples_y
    