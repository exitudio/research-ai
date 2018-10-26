from abc import ABC, abstractmethod
import numpy as np


class Activation(ABC):
    @abstractmethod
    def feed_forward(self, input):
        pass

    @abstractmethod
    def back_prop(self):
        pass


class Sigmoid(Activation):
    def feed_forward(self, input):
        self.output = 1.0/(1 + np.exp(-input))
        return self.output

    def back_prop(self):
        return self.output * (1.0 - self.output)


class Relu(Activation):
    def feed_forward(self, input):
        self.input = input
        output = np.maximum(0, input)
        return output

    def back_prop(self):
        return (self.input > 0) * 1


class NoActivation(Activation):
    def feed_forward(self, input):
        return input

    def back_prop(self):
        return 1
