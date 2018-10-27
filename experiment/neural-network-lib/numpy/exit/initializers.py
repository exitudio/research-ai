from abc import ABC, abstractmethod
import numpy as np

'''
tensorflow implementation
  https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/init_ops.py
Deeplearning.ai
 https://www.coursera.org/learn/deep-neural-network/lecture/RwqYe/weight-initialization-for-deep-networks
'''


class Initializer(ABC):
    @abstractmethod
    def __call__(self, num_input, num_output):
        pass


class Constant(Initializer):
    def __init__(self, value):
        self._value = value

    def __call__(self, num_input, num_output):
        return np.full((num_input, num_output), self._value)


class GlorotNormal(Initializer):
    def __call__(self, num_input, num_output):
        scale = 2/(num_input + num_output)
        stddev = np.sqrt(scale) / .87962566103423978
        return np.random.randn(num_input, num_output) * np.sqrt(stddev)


class GlorotUniform(Initializer):
    """The Glorot uniform initializer, also called Xavier uniform initializer.
      It draws samples from a uniform distribution within [-limit, limit]
      where `limit` is `sqrt(6 / (fan_in + fan_out))`
      where `fan_in` is the number of input units in the weight tensor
      and `fan_out` is the number of output units in the weight tensor.
      Reference: http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf
    """

    def __call__(self, num_input, num_output):
        scale = 2/(num_input + num_output)
        limit = np.sqrt(3.0 * scale)
        return np.random.uniform(-limit, limit, (num_input, num_output))


# if dtype is None:
#       dtype = self.dtype
#     scale = self.scale
#     scale_shape = shape
#     if partition_info is not None:
#       scale_shape = partition_info.full_shape
#     fan_in, fan_out = _compute_fans(scale_shape)
#     if self.mode == "fan_in":
#       scale /= max(1., fan_in)
#     elif self.mode == "fan_out":
#       scale /= max(1., fan_out)
#     else:
#       scale /= max(1., (fan_in + fan_out) / 2.)
#     if self.distribution == "normal" or self.distribution == "truncated_normal":
#       # constant taken from scipy.stats.truncnorm.std(a=-2, b=2, loc=0., scale=1.)
#       stddev = math.sqrt(scale) / .87962566103423978
#       return random_ops.truncated_normal(
#           shape, 0.0, stddev, dtype, seed=self.seed)
#     elif self.distribution == "untruncated_normal":
#       stddev = math.sqrt(scale)
#       return random_ops.random_normal(
#           shape, 0.0, stddev, dtype, seed=self.seed)
#     else:
#       limit = math.sqrt(3.0 * scale)
#       return random_ops.random_uniform(
#           shape, -limit, limit, dtype, seed=self.seed)
