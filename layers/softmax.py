import numpy as np

from layers.base_layer import BaseLayer


class Softmax(BaseLayer):

    def forward(self, input):
        input_shift = input - np.max(input)
        exp = np.exp(input_shift)
        return exp / np.sum(exp)

    def back(self, activation_theta, activation):
        return activation_theta

    def update_weights(self, input_batch, error_batch, learning_rate):
        return error_batch
