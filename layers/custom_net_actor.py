import numpy as np

from layers.cross_entropy import CrossEntropy


class CustomNetActor:

    def __init__(self, batch_size, learning_rate, value_model):
        self.layers = []
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.folds_number = 10
        self.loss_fn = CrossEntropy()
        self.value_model = value_model

    def add_layer(self, layer):
        self.layers.append(layer)

    def learn(self, inputs, target):
        inp = inputs[0:1, :]
        forward_activations = [inp]
        output = inp
        for layer in self.layers:
            output = layer.forward(output)
            forward_activations.append(output)

        q_value = self.value_model.predict_value(inp)
        action, reward = target[0]
        advantage = q_value if reward == 0 else reward
        q_vector = np.zeros(4)
        q_vector[action] = 1

        loss = self.loss_fn.loss(forward_activations.pop(), q_vector)
        theta = advantage * self.loss_fn.delta(output, q_vector)

        error = theta
        for layer in reversed(self.layers):
            input_batch = forward_activations.pop()
            error = layer.update_weights(input_batch, error, self.learning_rate)

    def predict(self, input_vector):
        for layer in self.layers:
            input_vector = layer.forward(input_vector)

        return input_vector
