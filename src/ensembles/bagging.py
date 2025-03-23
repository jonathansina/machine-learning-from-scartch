import numpy as np
from TreePackage import IdentificationTree
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score, accuracy_score 
import copy


class Bagging(object):
    def __init__(self, type: str = "classifier"):
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
        self.estimator = estimator
        self.number_of_estimators = number_of_estimators
        self.max_features = max_features
        self.max_samples = max_samples
        self.estimator_stack = []

    def fit(self, x_train: np.ndarray, y_train: np.ndarray, verbose: int = 2):
        if self.max_features is None:
            self.max_features = x_train.shape[1]
        elif self.max_features == "sqrt":
            self.max_features = int(np.sqrt(x_train.shape[1]))
        elif self.max_features == "log":
            self.max_features = int(np.log(x_train.shape[1]))

        x_samples, y_samples = self.bootstrap_aggregating(x_train, y_train)
        for i in range(self.number_of_estimators):
            tree = copy.deepcopy(self.estimator)
            tree.fit(x_samples[i], y_samples[i])
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

    def fit(self, x_train: np.ndarray, y_train: np.ndarray, verbose: int = 2) -> np.ndarray:
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
            tree.fit(x_train, residual)

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
        prediction = np.array([self.y_mean] * len(x))
        for tree in self.forest:
            prediction += np.array(tree.predict(x)) * self.learning_rate
        return prediction

    def residual_error(self, pred, y: np.ndarray) -> np.ndarray:
        return self.loss.derivative(pred, y)

    def plot_loss(self, errors: np.ndarray):
        plt.plot(range(self.number_of_estimators), errors)
        plt.xlabel('Number of Epochs')
        plt.ylabel('Cost')
        plt.title('Cost Function')
        plt.grid(True)
        plt.show()