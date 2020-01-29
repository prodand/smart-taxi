from torch import nn
import torch as tr


class DqnNetwork:

    def __init__(self, input_size):
        self.input_size = input_size
        self.model = nn.Sequential(
            nn.Conv1d(1, 4, input_size),
            nn.Sigmoid(),
        )

    def predict(self, data):
        return self.model(tr.from_numpy(data).float().reshape((1, 1, self.input_size))).detach().numpy().reshape(4)

    def fit(self, train_data, target):
        tensor_data = tr.from_numpy(train_data).float().reshape((train_data.shape[0], 1, self.input_size))
        tensor_target = tr.from_numpy(target)
        output = self.model(tensor_data)

        output.backward(tensor_target.reshape((output.shape[0], output.shape[1], 1)))

        self.model.zero_grad()

        learning_rate = 0.01
        for f in self.model.parameters():
            f.data.add_(f.grad.data * learning_rate)

        for i in range(len(train_data)):
            result = self.predict(train_data[i:i+1, :])
            i = 0