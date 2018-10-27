from abc import ABC, abstractmethod
import numpy as np
from .activations import NoActivation


class Layer(ABC):
    @abstractmethod
    def init_weights(self, num_input, optimizer):
        pass

    @abstractmethod
    def feed_forward(self):
        pass

    @abstractmethod
    def back_prop(self):
        pass


class Dense(Layer):
    def __init__(self, num_output, Activation_function=None):
        self._num_output = num_output
        self._activation_function = Activation_function(
        ) if Activation_function is not None else NoActivation()

    def init_weights(self, num_input, optimizer, initializer):
        self._num_input = num_input
        # init weights
        self._weights = initializer(num_input, self._num_output)
        self._bias = initializer(1, self._num_output)
        # init optimizer
        self._optimizer_w = optimizer.generate_optimizer(self._weights.shape)
        self._optimizer_b = optimizer.generate_optimizer(self._bias.shape)

    def feed_forward(self, input):
        z = np.dot(input, self._weights) + self._bias
        output = self._activation_function.feed_forward(z)
        self._input = input
        self._output = output
        return output

    def back_prop(self, last_derivative, learning_rate):
        dz = last_derivative * self._activation_function.back_prop()

        # it should be mean, but I don't know why not dividing by total is better
        # np.dot(self._input.T, dz)/self._num_input
        dw = np.dot(self._input.T, dz)
        db = np.sum(dz)  # np.mean(dz)
        # ------------------------------------------------------

        current_derivative = np.dot(dz, self._weights.T)
        self._weights -= learning_rate * self._optimizer_w.get_velocity(dw)
        self._bias -= learning_rate * self._optimizer_b.get_velocity(db)

        return current_derivative

    @property
    def num_output(self):
        return self._num_output
