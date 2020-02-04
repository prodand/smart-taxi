import numpy as np


class CustomNet:

    def __init__(self, batch_size, learning_rate, learning_rate_decay=0.005):
        self.layers = []
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.folds_number = 10

    def add_layer(self, layer):
        self.layers.append(layer)

    def learn(self, inputs, target):
        layers_cache = list()
        for layer in self.layers:
            layers_cache.append(list())

        inp = inputs[0:1, :]
        forward_activations = [inp]
        output = inp
        for layer in self.layers:
            output = layer.forward(output)
            forward_activations.append(output)

        last = forward_activations.pop()
        reward = target[0]
        q_value_new = self.predict(inputs[0]) if reward != -1 else 0
        theta = (reward + 0.9 * q_value_new) - last
        # theta = reward - last

        error = np.array(theta)
        for layer in reversed(self.layers):
            input_batch = forward_activations.pop()
            error = layer.update_weights(input_batch, error, self.learning_rate)

    def predict(self, input):
        for layer in self.layers:
            input = layer.forward(input)

        return input
