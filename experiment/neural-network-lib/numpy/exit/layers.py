from abc import ABC, abstractmethod
import numpy as np
from .activations import NoActivation


class Layer(ABC):
    @abstractmethod
    def init_weights(self, num_input):
        pass

    @abstractmethod
    def feed_forward(self):
        pass

    @abstractmethod
    def back_prob(self):
        pass


class Dense(Layer):
    def __init__(self, num_output, Activation_function=None):
        self._num_output = num_output
        self.activation_function = Activation_function(
        ) if Activation_function is not None else NoActivation()

    def init_weights(self, num_input):
        self._num_input = num_input
        self._weights = np.full((num_input, self._num_output), 0.1) #np.random.rand(num_input, self._num_output)
        self._bias = np.full((1, self._num_output), 0.1) #np.random.rand(1, self._num_output)

    def feed_forward(self, input):
        z = np.dot(input, self._weights) + self._bias
        output = self.activation_function.feed_forward(z)
        self._input = input
        self._output = output
        return output

    def back_prob(self, last_derivative, learning_rate):
        dz = last_derivative * self.activation_function.back_prob()

        # it should be mean, but I don't know why not dividing by total is better
        dw = np.dot(self._input.T, dz) # np.dot(self._input.T, dz)/self._num_input
        db = np.sum(dz) #np.mean(dz)
        # ------------------------------------------------------

        current_derivative = np.dot(dz, self._weights.T)  # X input
        self._weights -= learning_rate*dw
        self._bias -= learning_rate*db

        return current_derivative

    @property
    def num_output(self):
        return self._num_output
