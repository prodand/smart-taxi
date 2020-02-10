import numpy as np

from layers.base_layer import BaseLayer


class FullyConnected(BaseLayer):

    def __init__(self, input_size, output_size, weights=None, bias=None, sigm=False):
        self.input_size = input_size
        self.output_size = output_size
        self.sigm = sigm
        if weights is None:
            self.weights = np.random.normal(0.0, 0.8, (input_size, output_size)) * (2 / np.sqrt(input_size))
        else:
            self.weights = weights.reshape((input_size, output_size))
        if bias is None:
            self.bias = np.random.normal(0.0, 0.5, (1, output_size)) * (2 / np.sqrt(output_size))
        else:
            self.bias = bias.reshape((1, output_size))

    def forward(self, image_vector):
        layer_out = np.dot(image_vector, self.weights) + self.bias
        if self.sigm:
            return self.relu(layer_out)
        return self.relu(layer_out)

    def back(self, activation_theta, layer_input):
        return np.dot(self.sigmoid_(layer_input, activation_theta), self.weights.T)

    def sigmoid_(self, layer_input, theta):
        if self.sigm:
            z = self.forward(layer_input)
            return theta * (z * (1 - z))
        return theta

    def update_weights(self, input_batch, error_batch, learning_rate):
        input_error = self.back(error_batch, input_batch)
        linear_error = self.sigmoid_(input_batch, error_batch)
        derivative_weights = np.dot(input_batch.T, linear_error)
        batch_size = input_batch.shape[0]
        self.weights = self.weights + learning_rate * (derivative_weights / batch_size)
        self.bias = self.bias + learning_rate * (linear_error / batch_size)
        return input_error

    def relu(self, image):
        return np.maximum(image, 0)

    def sigmoid(self, input):
        return 1 / (1 + np.exp(-input))

    def save(self, folder: str):
        weights_file_name = "/fc_weights_%s" % self.input_size
        np.save(folder + weights_file_name, self.weights)
        bias_file_name = "/fc_bias_%s" % self.output_size
        np.save(folder + bias_file_name, self.bias)
        return "%s,%s" % (weights_file_name, bias_file_name)

    @staticmethod
    def load(folder, files):
        parts = files.split(",")
        weights = np.load(folder + parts[0] + ".npy")
        bias = np.load(folder + parts[1] + ".npy")
        return FullyConnected(weights.shape[1], weights.shape[0],
                              weights, bias)
