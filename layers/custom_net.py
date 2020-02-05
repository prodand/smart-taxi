import numpy as np


class CustomNet:

    def __init__(self, batch_size, learning_rate):
        self.layers = []
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.folds_number = 10

    def add_layer(self, layer):
        self.layers.append(layer)

    def learn(self, inputs, target):
        inp = inputs[0:1, :]
        forward_activations = [inp]
        output = inp
        for layer in self.layers:
            output = layer.forward(output)
            forward_activations.append(output)

        last = forward_activations.pop()
        reward = target[0]
        q_value_new = self.predict(inputs[1]) if reward != -1 and reward != 10.0 else 0
        theta = last - (reward + 0.9 * q_value_new)

        error = np.array(theta)
        for layer in reversed(self.layers):
            input_batch = forward_activations.pop()
            error = layer.update_weights(input_batch, error, self.learning_rate)

    def predict(self, input_vector):
        for layer in self.layers:
            input_vector = layer.forward(input_vector)

        return input_vector
