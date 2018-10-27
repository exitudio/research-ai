from abc import ABC, abstractmethod
import numpy as np
from .constants import EPSILON


class Optimizer(ABC):
    @abstractmethod
    def get_velocity(self, d):
        pass

    @abstractmethod
    def generate_optimizer(self, shape):
        pass


class GradientDescent(Optimizer):
    def get_velocity(self, d):
        return d

    def generate_optimizer(self, shape):
        return GradientDescent()


class Momentum(Optimizer):
    # In coursera, prof Andrew suggest beta=.9 ~ average out around 10 pass data
    def __init__(self, beta=0.9, shape=None):
        self._beta = beta

        if shape is not None:
            self._velocity = np.zeros(shape)

    def get_velocity(self, d):
        self._velocity = self._beta * self._velocity + (1-self._beta) * d
        return self._velocity

    def generate_optimizer(self, shape):
        return Momentum(self._beta, shape)


class RMSProp(Optimizer):
    def __init__(self, beta=0.999, shape=None):
        self._beta = beta

        if shape is not None:
            self._s = np.zeros(shape)

    def get_velocity(self, d):
        self._s = self._beta * \
            self._s + (1-self._beta) * d**2
        return d/ (np.sqrt(self._s)+EPSILON)

    def generate_optimizer(self, shape):
        return RMSProp(self._beta, shape)


class Adam(Optimizer):
    def __init__(self, beta1=0.9, beta2=0.999, shape=None):
        self._beta1 = beta1
        self._beta2 = beta2

        if shape is not None:
            self._v = np.zeros(shape)
            self._s = np.zeros(shape)

    def get_velocity(self, d):
        self._v = self._beta1 * self._v + (1-self._beta1) * d
        self._s = self._beta2 * \
            self._s + (1-self._beta2) * d**2
        return self._v/ (np.sqrt(self._s)+EPSILON)

    def generate_optimizer(self, shape):
        return Adam(self._beta1, self._beta2, shape)
