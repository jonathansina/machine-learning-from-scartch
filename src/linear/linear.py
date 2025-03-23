import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

from Losses import *
from Optimizers import *


class Regularizer(object):
    def __init__(self, type: str, lamda: float = 1, alpha: float = 0.8):
        if type == 'l1' or type == 'l2' or type == 'l1l2':
            self.type = type
        else:
            raise ValueError('Invalid regularizer!')
        self.lamda = lamda
        self.alpha = alpha

    def __call__(self, weights: np.ndarray):
        if self.type == 'l1':
            return self.lamda * np.sum(np.abs(weights))
        elif self.type == 'l2':
            return self.lamda * np.sum(np.square(weights))
        else:
            return self.lamda * (self.alpha * np.sum(np.abs(weights)) + (1 - self.alpha) * np.sum(np.square(weights)))

    def derivative(self, weights: np.ndarray):
        if self.type == 'l1':
            return self.lamda * np.sign(weights)
        elif self.type == 'l2':
            return self.lamda * 2 * weights
        else:
            return self.lamda * ((self.alpha * np.sign(weights)) + 1 - self.alpha * 2 * weights)


class LinearModel:
    def __init__(self):
        self.mini_batches_data = None
        self.loss = None
        self.optimizer = None
        self.regularizer = None
        self.weight_matrix = np.array([])
        self.bias = np.array([])
        self.cost = []
        self.batch_size = 1

    def compile(self, optimizer: classmethod, loss: classmethod, regularizer: classmethod = None):
        if optimizer == 'adagrad':
            self.optimizer = AdaGrad()
        elif optimizer == 'rmsprop':
            self.optimizer = RMSProp()
        elif optimizer == 'adam':
            self.optimizer = ADAM()
        elif optimizer == 'sgd' or optimizer == 'gradient-descent':
            self.optimizer = SGD()
        elif optimizer == 'newton-method':
            self.optimizer = NewtonMethod()
        elif isinstance(optimizer, (RMSProp, ADAM, AdaGrad, SGD, NewtonMethod)):
            self.optimizer = optimizer
        else:
            raise ValueError('Invalid optimizer!')

        self.init_loss_functions(loss)

        if regularizer == "l1":
            self.regularizer = Regularizer(type="l1")
        elif regularizer == "l2":
            self.regularizer = Regularizer(type="l2")
        elif isinstance(regularizer, Regularizer):
            self.regularizer = regularizer
        elif regularizer is None:
            raise NotImplementedError("Subclasses must implement this method")
        else:
            raise ValueError('Invalid regularizer!')

    def init_loss_functions(self, loss):
        raise NotImplementedError("Subclasses must implement this method")

    def train(self, x: np.ndarray, y: np.ndarray, epochs: int, batch_size: int = 1, verbose: int = 2):
        self.init_parameters(x, y)
        if (self.optimizer.name == "gradient-descent" and batch_size == 1) or self.optimizer.name == "newton-method":
            self.batch_size = x.shape[0]
        else:
            self.batch_size = batch_size

        counter = 0
        self.mini_batches_data = self.create_mini_batches(x, y, self.batch_size)
        if self.batch_size == 1:
            self.mini_batches_data.pop()

        while counter < epochs:
            errors = []
            counter += 1
            for i in range(len(self.mini_batches_data)):
                gradients = [[], []]
                for j in range(self.mini_batches_data[i][0].shape[0]):
                    # calculate the output of the model
                    y_out = self.predict(self.mini_batches_data[i][0][j])

                    # calculate loss of the output
                    errors.append(self.loss(y_out, self.mini_batches_data[i][1][j]))

                    # optimization
                    delta_gradient, gradient = self.optimize(y_out, i, j)

                    if self.batch_size != 1:
                        gradients[0].append(delta_gradient)
                        gradients[1].append(gradient)
                    else:
                        self.update_parameters(delta_gradient, gradient, self.optimizer.name, i)

                if self.batch_size != 1:
                    delta_gradient = sum(gradients[0]) / len(gradients[0])
                    gradient = sum(gradients[1]) / len(gradients[1])
                    self.update_parameters(delta_gradient, gradient, self.optimizer.name, i)

            self.cost.append(sum(errors) / len(errors))
            if verbose <= 2:
                print("-------------------[EPOCH {}/{}]---------------------".format(counter, epochs))
                print('Error :', self.cost[-1])

        if verbose <= 1:
            self.plot_loss(counter)
        if verbose <= 0:
            print('Optimization Finished')

    def init_parameters(self, x: np.ndarray, y: np.ndarray):
        self.weight_matrix = np.random.uniform(-1, 1, size=(y.shape[1], x.shape[1]))
        self.bias = np.random.uniform(-1, 1, size=(y.shape[1]))

    def predict(self, x: np.ndarray):
        raise NotImplementedError("Subclasses must implement this method")

    def plot_loss(self, counter):
        plt.plot(range(counter), self.cost)
        plt.xlabel('Number of Epochs')
        plt.ylabel('Cost')
        plt.title('Cost Function')
        plt.grid(True)

    @staticmethod
    def create_mini_batches(x, y, batch_size):
        mini_batches = []
        data = np.hstack((x, y))
        np.random.shuffle(data)
        n_minibatches = data.shape[0] // batch_size
        i = 0

        for i in range(n_minibatches + 1):
            mini_batch = data[i * batch_size:(i + 1) * batch_size, :]
            x_mini = mini_batch[:, :-1]
            y_mini = mini_batch[:, -1].reshape((-1, 1))
            mini_batches.append((x_mini, y_mini))
        if data.shape[0] % batch_size != 0:
            mini_batch = data[i * batch_size:data.shape[0]]
            x_mini = mini_batch[:, :-1]
            y_mini = mini_batch[:, -1].reshape((-1, 1))
            mini_batches.append((x_mini, y_mini))
        return mini_batches

    def optimize(self, y_out: np.ndarray, var_i: int, var_j: int):
        penalty_term = 0
        if self.regularizer is not None:
            penalty_term = self.regularizer.derivative(self.weight_matrix)

        delta_gradient = self.loss.derivative(y_out, self.mini_batches_data[var_i][1][var_j])
        gradient = (delta_gradient * self.mini_batches_data[var_i][0][var_j] + penalty_term)

        return delta_gradient, gradient

    def update_parameters(self, delta_gradient: np.ndarray, gradient: np.ndarray, optimizer: str, i: int = 0):
        if optimizer == "adam" or optimizer == "sgd":
            self.weight_matrix, self.bias = self.optimizer(self.weight_matrix, self.bias, gradient, delta_gradient, i)
        elif optimizer == "newton-method":
            self.weight_matrix, self.bias = self.optimizer(self.weight_matrix, self.bias, gradient, delta_gradient)
        else:
            self.weight_matrix, self.bias = self.optimizer(self.weight_matrix, self.bias, gradient, delta_gradient)


class LinearRegression(LinearModel):
    def __init__(self):
        super().__init__()

    def init_loss_functions(self, loss):
        if loss == 'mse':
            self.loss = MeanSquaredError()
        elif loss == 'mae':
            self.loss = MeanAbsoluteError()
        elif loss == 'huber':
            self.loss = Huber()
        elif isinstance(loss, (MeanSquaredError, MeanAbsoluteError, Huber)):
            self.loss = loss
        else:
            raise ValueError('Invalid loss function!')

    def predict(self, x: np.ndarray):
        if x.T.shape != x.shape and x.shape[0] != 1:
            return np.array([self.predict(x[i]) for i in range(x.shape[0])]).reshape(x.shape[0], 1)
        y_out = self.weight_matrix.dot(x) + self.bias
        return y_out


class LogisticRegression(LinearModel):
    def __init__(self):
        super().__init__()

    def init_loss_functions(self, loss='log_loss'):
        if loss == 'log_loss':
            self.loss = LogLoss()
        if loss == 'binary_crossentropy':
            self.loss = BinaryCrossEntropy()
        elif isinstance(loss, (LogLoss, BinaryCrossEntropy)):
            self.loss = loss
        else:
            raise ValueError('Invalid loss function!')

    def predict(self, x: np.ndarray):
        if x.T.shape != x.shape and x.shape[0] != 1:
            return np.array([self.predict(x[i]) for i in range(x.shape[0])]).reshape(x.shape[0], 1)
        y_out = 1 / (1 + np.exp(-(self.weight_matrix.dot(x) + self.bias)))
        return y_out

    def classify(self, x: np.ndarray, threshold: float):
        predicted = self.predict(x)
        return [0 if predicted[i] < threshold else 1 for i in range(predicted.shape[0])]


class SVM(LinearModel):
    def __init__(self):
        super().__init__()

    def init_loss_functions(self, loss='hinge'):
        if loss == 'hinge':
            self.loss = Hinge()
        elif isinstance(loss, Hinge):
            self.loss = loss
        else:
            raise ValueError('Invalid loss function!')

    def predict(self, x: np.ndarray):
        if x.T.shape != x.shape and x.shape[0] != 1:
            return np.array([self.predict(x[i]) for i in range(x.shape[0])]).reshape(x.shape[0], 1)
        y_out = np.sign(self.weight_matrix.dot(x) + self.bias)
        return y_out


def plot_accuracy(x_train, y_train, x_test, y_test, loss, optimizer, loss_name, ranges):
    scores = []
    interval = [i for i in range(10, ranges + 1, 10)]
    for i in interval:
        model = LinearRegression()
        model.compile(
            optimizer=optimizer,
            loss=loss
        )
        model.train(
            x_train,
            y_train,
            epochs=i,
            verbose=False
        )
        scores.append(accuracy(x_test, y_test, model))

    plt.plot(interval, scores, 'o-')
    plt.title(f"R2 Scores VS. Epochs ({loss_name} loss)")
    plt.xlabel("Number of Epochs")
    plt.ylabel("R2 Score")
    plt.grid(True)
    return interval, scores

def plot_accuracy(x_train, y_train, x_test, y_test, loss, optimizer, loss_name, ranges):
    scores = []
    interval = [i for i in range(10, ranges + 1, 10)]
    for i in interval:
        model = LinearRegression()
        model.compile(
            optimizer=optimizer,
            loss=loss
        )
        model.train(
            x_train,
            y_train,
            epochs=i,
            verbose=False
        )
        scores.append(accuracy(x_test, y_test, model))

    plt.plot(interval, scores, 'o-')
    plt.title(f"R2 Scores VS. Epochs ({loss_name} loss)")
    plt.xlabel("Number of Epochs")
    plt.ylabel("R2 Score")
    plt.grid(True)
    return interval, scores


def accuracy(x_test, y_test: np.ndarray, model):
    y_pred = model.predict(x_test)
    return r2_score(y_test, y_pred)


