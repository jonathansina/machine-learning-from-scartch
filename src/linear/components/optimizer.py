import numpy as np


class AdaGrad(object):
    def __init__(self, learning_rate: float = 0.001, epsilon: float = 1e-8):
        self.learning_rate = learning_rate
        self.name = 'adagrad'
        self.z_weights = 0
        self.z_bias = 0
        self.epsilon = epsilon

    def __call__(self, weights: np.ndarray, bias: np.ndarray, gradient: np.ndarray, delta_gradient: np.ndarray):
        self.z_weights = self.z_weights + np.square(gradient)
        self.z_bias = self.z_bias + np.square(delta_gradient)

        weights = weights + (self.learning_rate / (np.sqrt(self.z_weights + self.epsilon))) * gradient
        bias = bias + (self.learning_rate / (np.sqrt(self.z_bias + self.epsilon))) * delta_gradient
        return weights, bias


class RMSProp(object):
    def __init__(self, learning_rate: float = 0.001, beta: float = 0.9, epsilon: float = 1e-8):
        self.learning_rate = learning_rate
        self.name = 'rmsprop'
        self.z_weights = 0
        self.z_bias = 0
        self.beta = beta
        self.epsilon = epsilon

    def __call__(self, weights: np.ndarray, bias: np.ndarray, gradient: np.ndarray, delta_gradient: np.ndarray):
        self.z_weights = self.beta * self.z_weights + (1 - self.beta) * np.square(gradient)
        self.z_bias = self.beta * self.z_bias + (1 - self.beta) * np.square(delta_gradient)

        weights = weights + (self.learning_rate / (np.sqrt(self.z_weights + self.epsilon))) * gradient
        bias = bias + (self.learning_rate / (np.sqrt(self.z_bias + self.epsilon))) * delta_gradient
        return weights, bias


class ADAM(object):
    def __init__(self, learning_rate: float = 0.001, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8):
        self.learning_rate = learning_rate
        self.name = 'adam'
        self.z_weights = 0
        self.z_bias = 0
        self.m_weights = 0
        self.m_bias = 0
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    def __call__(self, weights: np.ndarray, bias: np.ndarray, gradient: np.ndarray, delta_gradient: np.ndarray, i: int):
        self.z_weights = self.beta2 * self.z_weights + (1 - self.beta2) * np.square(gradient)
        self.z_bias = self.beta2 * self.z_bias + (1 - self.beta2) * np.square(delta_gradient)
        self.m_weights = self.beta2 * self.m_weights + (1 - self.beta1) * gradient
        self.m_bias = self.beta2 * self.m_bias + (1 - self.beta1) * delta_gradient

        z_weights_hat = np.divide(self.z_weights, 1 - np.power(self.beta2, i + 1))
        z_bias_hat = np.divide(self.z_bias, 1 - np.power(self.beta2, i + 1))
        m_weights_hat = np.divide(self.m_weights, 1 - np.power(self.beta1, i + 1))
        m_bias_hat = np.divide(self.m_bias, 1 - np.power(self.beta1, i + 1))

        weights = weights + (self.learning_rate / (np.sqrt(z_weights_hat + self.epsilon))) * m_weights_hat
        bias = bias + (self.learning_rate / (np.sqrt(z_bias_hat + self.epsilon))) * m_bias_hat
        return weights, bias


class SGD(object):
    def __init__(self, learning_rate: float = 0.001, momentum: float = 0.0, nesterov: bool = False, tau: int = 200):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.nesterov = nesterov
        self.w_velocity = 0
        self.b_velocity = 0
        self.name = 'sgd'
        self.tau = tau
        self.learning_rate_0 = self.learning_rate

    def __call__(self, weights: np.ndarray, bias: np.ndarray, gradient: np.ndarray, delta_gradient: np.ndarray, i: int):
        alpha = i / self.tau
        if self.momentum == 0:
            weights = weights + self.learning_rate * gradient
            bias = bias + self.learning_rate * delta_gradient
            if i <= self.tau:
                self.learning_rate = (1 - alpha) * self.learning_rate_0 + alpha * 0.01 * self.learning_rate_0
        else:
            self.velocity = self.momentum * self.w_velocity + self.learning_rate * gradient
            self.b_velocity = self.momentum * self.w_velocity + self.learning_rate * delta_gradient
            if not self.nesterov:
                weights = weights + self.velocity
                bias = bias + self.b_velocity
            else:
                weights = weights + self.learning_rate * self.velocity + self.learning_rate * gradient
                bias = bias + self.learning_rate * self.b_velocity + self.learning_rate * delta_gradient
        return weights, bias


class NewtonMethod(object):
    def __init__(self):
        self.name = 'newton-method'

    def __call__(self, weights: np.ndarray, bias: np.ndarray, gradient: np.ndarray, delta_gradient: np.ndarray,
                 hessian: np.matrix):
        hessian_inverse = hessian.getI()
        weights = weights + hessian_inverse[0] * gradient
        bias = bias + hessian_inverse[1] * delta_gradient
        return weights, bias