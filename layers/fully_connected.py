import numpy as np

from layers.base_layer import BaseLayer


class FullyConnected(BaseLayer):

    def __init__(self, input_size, output_size, weights=None, bias=None, sigm = False):
        self.input_size = input_size
        self.output_size = output_size
        self.sigm = sigm
        if weights is None:
            self.weights = np.random.normal(0.0, 0.8, (output_size, input_size)) * (2 / np.sqrt(input_size))
        else:
            self.weights = weights
        if bias is None:
            self.bias = np.random.normal(0.0, 0.5, (output_size, 1)) * (2 / np.sqrt(output_size))
        else:
            self.bias = bias

    def forward(self, image_vector):
        layer_out = np.dot(image_vector, self.weights.T) + self.bias.T
        if self.sigm:
            return self.sigmoid(layer_out)
        return layer_out

    def back(self, activation_theta, layer_input):
        prev_layer_error = np.dot(activation_theta.T, self.weights)
        if self.sigm:
            return prev_layer_error * (layer_input * (1 - layer_input))
        return prev_layer_error

    def update_weights(self, layer_cache, learning_rate):
        images_matrix = []
        activation_theta_matrix = []
        derived_biases = np.zeros(self.bias.shape)
        for (image_vector, activation_theta) in layer_cache:
            images_matrix = np.column_stack((images_matrix, image_vector)) \
                if len(images_matrix) > 0 else image_vector
            activation_theta_matrix = np.column_stack((activation_theta_matrix, activation_theta)) \
                if len(activation_theta_matrix) > 0 else activation_theta
            derived_biases += activation_theta

        derivative_weights = activation_theta_matrix.dot(images_matrix.T)
        self.weights = self.weights - learning_rate * (derivative_weights / len(layer_cache))
        self.bias = self.bias - learning_rate * (derived_biases / len(layer_cache))

    def relu(self, image):
        return np.maximum(image, 0)

    def sigmoid(self, input):
        return 1/(1 + np.exp(-input))

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
