from .optimizers import GradientDescent
from .initializers import GlorotNormal

class NeuralNetwork:
    def __init__(self, optimizer=GradientDescent(), initializer=GlorotNormal()):
        self._network = []
        self._optimizer = optimizer
        self._initializer = initializer

    def sequence(self, num_input_feature, *args):
        self._network = args
        num_input = num_input_feature
        for _, layer in enumerate(self._network):
            layer.init_weights(num_input, self._optimizer, self._initializer)
            num_input = layer.num_output

    def train(self, input, expected_output, Loss_function, learning_rate=0.1):
        # feed forward
        output_from_layer = input
        for _, layer in enumerate(self._network):
            output_from_layer = layer.feed_forward(output_from_layer)

        # loss
        loss_function = Loss_function(expected_output, output_from_layer)

        # back prop
        derivative = loss_function.get_derivative_loss()
        for _, layer in reversed(list(enumerate(self._network))):
            derivative = layer.back_prop(derivative, learning_rate)
        return {
            'output_from_layer': output_from_layer,
            'loss': loss_function.get_loss()
        }

    def predict(self, input):
        output_from_layer = input
        for _, layer in enumerate(self._network):
            output_from_layer = layer.feed_forward(output_from_layer)
        return output_from_layer
