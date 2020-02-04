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

        for index in range(len(inputs) - 1):
            input = inputs[index]
            forward_activations = [input]
            output = input
            for layer in self.layers:
                output = layer.forward(output)
                forward_activations.append(output)

            last = forward_activations.pop()
            reward = target[index]
            q_value_new = self.predict(inputs[index + 1]) if reward != -1 else 0
            theta = (reward + 0.9 * q_value_new) - last

            activation_thetas = [theta]
            activation = last
            a_idx = len(self.layers) - 1
            for layer in reversed(self.layers[1:]):
                activation = forward_activations[a_idx]
                theta = layer.back(theta, activation)
                a_idx -= 1
                activation_thetas.append(theta)

            layer_index = 0
            for (saved_image, theta) in zip(forward_activations, reversed(activation_thetas)):
                layers_cache[layer_index].append((saved_image, theta))
                layer_index += 1

        for (layer, cache) in zip(self.layers, layers_cache):
            layer.update_weights(cache, self.learning_rate)

    def predict(self, input):
        for layer in self.layers:
            input = layer.forward(input)

        return input
