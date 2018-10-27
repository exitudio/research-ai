from abc import ABC, abstractmethod
import numpy as np
from .constants import EPSILON


class Loss(ABC):
    def __init__(self, expected_output, predict_output):
        self._expected_output = expected_output
        self._predict_output = predict_output

    @abstractmethod
    def get_loss(self):
        pass

    @abstractmethod
    def get_derivative_loss(self):
        pass


class CrossEntropy(Loss):
    def __init__(self, expected_output, predict_output):
        super().__init__(expected_output, predict_output)

    def get_loss(self):
        loss = np.mean(-(self._expected_output*np.log(self._predict_output) +
                         (1-self._expected_output) * np.log(1-self._predict_output+EPSILON)))
        return loss

    def get_derivative_loss(self):
        dloss = (-(self._expected_output / (self._predict_output+EPSILON)) +
                (1-self._expected_output)/(1-self._predict_output+EPSILON))
        return dloss


class L2(Loss):
    def __init__(self, expected_output, predict_output):
        super().__init__(expected_output, predict_output)

    def get_loss(self):
        return np.mean(np.square(self._predict_output - self._expected_output))

    def get_derivative_loss(self):
        return 2*(self._predict_output - self._expected_output)
