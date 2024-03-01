import numpy as np
from TreePackage import IdentificationTree
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score, accuracy_score 
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
                    if self.type == "classifier":
                        training_error = accuracy_score(y_samples[i], y_pred)
                    else:
                        training_error = r2_score(y_samples[i], y_pred)
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
    
    
class GradientBoostedRegressionTree(object):
    """
    A class for Gradient Boosted Regression Trees.

    Parameters
    ----------
    number_of_estimators : int, optional (default=10)
        The number of estimators in the ensemble.

    max_depth : int, optional (default=3)
        The maximum tree depth.

    max_features : int or str, optional (default="sqrt")
        The number of features to consider when looking for the best split.
        If "sqrt", then sqrt(number_of_features) are considered.

    loss : str, optional (default="squared-loss")
        The loss function to be used for optimization.
        Can be "squared-loss", "absolute-loss", or "huber".

    learning_rate : float, optional (default=0.1)
        The learning rate for each iteration.

    Attributes
    ----------
    y_mean : float
        The mean of the target variable.

    learning_rate : float
        The learning rate used in training.

    loss : LossFunction
        The loss function used in training.

    max_features : int or str
        The maximum number of features used in tree pruning.

    max_depth : int
        The maximum tree depth used in training.

    number_of_estimators : int
        The number of estimators in the ensemble.

    forest : list of IdentificationTree
        The list of trees in the ensemble.

    Methods
    -------
    compile(self, number_of_estimators=10, max_depth=3, max_features="sqrt",
            loss="squared-loss", learning_rate=0.1)
        Compiles the model by setting the hyperparameters.

    train(self, x_train, y_train, verbose=2)
        Trains the model on the given training data.

    predict(self, x)
        Predicts the target values for the given data.

    residual_error(self, pred, y)
        Calculates the residual error for a given prediction and target values.

    plot_loss(self, errors)
        Plots the loss function over the number of iterations.

    """

    def __init__(self):
        self.y_mean = 0
        self.learning_rate = None
        self.loss = None
        self.max_features = None
        self.max_depth = None
        self.number_of_estimators = None
        self.forest = []

    def compile(self, number_of_estimators: int = 10, max_depth: int = 3, max_features: int | str = "sqrt",
                loss: str = "squared-loss", learning_rate: int = 0.1):
        """
        Compiles the model by setting the hyperparameters.

        Parameters
        ----------
        number_of_estimators : int, optional (default=10)
            The number of estimators in the ensemble.

        max_depth : int, optional (default=3)
            The maximum tree depth.

        max_features : int or str, optional (default="sqrt")
            The number of features to consider when looking for the best split.
            If "sqrt", then sqrt(number_of_features) are considered.

        loss : str, optional (default="squared-loss")
            The loss function to be used for optimization.
            Can be "squared-loss", "absolute-loss", or "huber".

        learning_rate : float, optional (default=0.1)
            The learning rate for each iteration.

        """
        self.learning_rate = learning_rate
        self.number_of_estimators = number_of_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        if loss == "squared-loss":
            self.loss = MeanSquaredError()
        elif loss == "absolute-loss":
            self.loss = MeanAbsoluteError()
        elif loss == "huber":
            self.loss = Huber()
        else:
            raise ValueError("Invalid loss!")

    def train(self, x_train: np.ndarray, y_train: np.ndarray, verbose: int = 2) -> np.ndarray:
        """
        Trains the model on the given training data.

        Parameters
        ----------
        x_train : np.ndarray
            The training data features.

        y_train : np.ndarray
            The training data target values.

        verbose : int, optional (default=2)
            The verbosity level.

        Returns
        -------
        np.ndarray
            The list of errors for each iteration.

        """
        self.y_mean = np.mean(y_train)
        prediction = (np.ones(y_train.shape[0]) * self.y_mean).reshape(-1, 1)
        errors = []
        for i in range(self.number_of_estimators):
            residual = self.residual_error(prediction, y_train).reshape(-1, 1)

            tree = IdentificationTree(type="regressor")
            tree.compile(
                max_depth=self.max_depth,
                max_features=self.max_features,
                impurity_function="mse"
            )
            tree.train(x_train, residual)

            predict = tree.predict(x_train).reshape(-1, 1)
            prediction += predict * self.learning_rate
            errors.append(self.loss(prediction, y_train))
            self.forest.append(tree)

            if verbose <= 1:
                print(f"-------------------[Iteration {i}/{self.number_of_estimators}]---------------------")
                print('Error :', errors[-1])

        if verbose <= 0:
            self.plot_loss(np.array(errors))
        return np.array(errors)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predicts the target values for the given data.

        Parameters
        ----------
        x : np.ndarray
            The data features.

        Returns
        -------
        np.ndarray
            The predicted target values.

        """
        prediction = np.array([self.y_mean] * len(x))
        for tree in self.forest:
            prediction += np.array(tree.predict(x)) * self.learning_rate
        return prediction

    def residual_error(self, pred, y: np.ndarray) -> np.ndarray:
        """
        Calculates the residual error for a given prediction and target values.

        Parameters
        ----------
        pred : np.ndarray
            The predicted target values.

        y : np.ndarray
            The target values.

        Returns
        -------
        np.ndarray
            The residual error.

        """
        return self.loss.derivative(pred, y)

    def plot_loss(self, errors: np.ndarray):
        """
        Plots the loss function over the number of iterations.

        Parameters
        ----------
        errors : np.ndarray
            The list of errors for each iteration.

        """
        plt.plot(range(self.number_of_estimators), errors)
        plt.xlabel('Number of Epochs')
        plt.ylabel('Cost')
        plt.title('Cost Function')
        plt.grid(True)
        plt.show()