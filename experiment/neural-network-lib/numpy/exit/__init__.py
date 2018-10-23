class NeuralNetwork:
    def __init__(self):
        self.network = []

    def sequence(self, num_input_feature, *args):
        self.network = args
        num_input = num_input_feature
        for _, layer in enumerate(self.network):
            layer.init_weights(num_input)
            num_input = layer.num_output

    def train(self, input, expected_output, Loss_function, learning_rate=0.1):
        # feed forward
        output_from_layer = input
        for _, layer in enumerate(self.network):
            output_from_layer = layer.feed_forward(output_from_layer)

        # loss
        loss_function = Loss_function(expected_output, output_from_layer)

        # back prop
        derivative = loss_function.get_derivative_loss()
        for _, layer in reversed(list(enumerate(self.network))):
            derivative = layer.back_prob(derivative, learning_rate)
        return {
            'output_from_layer': output_from_layer,
            'loss': loss_function.get_loss()
        }

    def predict(self, input):
        output_from_layer = input
        for _, layer in enumerate(self.network):
            output_from_layer = layer.feed_forward(output_from_layer)
        return output_from_layer
