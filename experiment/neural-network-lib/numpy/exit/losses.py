from abc import ABC, abstractmethod
import numpy as np
from .constants import EPSILON


class Loss(ABC):
    def __init__(self, expected_output, predict_output):
        self.expected_output = expected_output
        self.predict_output = predict_output

    @abstractmethod
    def get_loss(self):
        pass

    @abstractmethod
    def get_derivative_loss(self):
        pass


class Cross_entropy(Loss):
    def __init__(self, expected_output, predict_output):
        super().__init__(expected_output, predict_output)

    def get_loss(self):
        loss = np.mean(-(self.expected_output*np.log(self.predict_output) +
                         (1-self.expected_output) * np.log(1-self.predict_output+EPSILON)))
        return loss

    def get_derivative_loss(self):
        dloss = (-(self.expected_output / (self.predict_output+EPSILON)) +
                (1-self.expected_output)/(1-self.predict_output+EPSILON))
        return dloss


class L2(Loss):
    def __init__(self, expected_output, predict_output):
        super().__init__(expected_output, predict_output)

    def get_loss(self):
        return np.mean(np.square(self.predict_output - self.expected_output))

    def get_derivative_loss(self):
        return 2*(self.predict_output - self.expected_output)
