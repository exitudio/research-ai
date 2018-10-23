from abc import ABC, abstractmethod
import numpy as np


class Optimizer(ABC):
  def __init__(self, beta):
    self.beta = beta
  
  @abstractmethod
  def get_velocity(self, d):
    pass


class GradientDescent(Optimizer):
  def __init__(self, beta):
    super().__init__(beta)

  def get_velocity(self, d):
    return d