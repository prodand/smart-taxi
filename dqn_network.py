import numpy as np
from torch import nn
import torch as tr


class DqnNetwork:

    def __init__(self, input_size):
        self.input_size = input_size
        self.model = nn.Sequential(
            nn.Conv1d(1, 4, input_size),
            nn.Sigmoid(),
            # nn.Softmax(dim=1)
        )
        self.value_model = nn.Sequential(
            nn.Linear(input_size, 4),
            nn.Sigmoid(),
            nn.Linear(4, 1),
        )

    def predict(self, data):
        return self.model(tr.from_numpy(data).float().reshape((1, 1, self.input_size))).detach().numpy().reshape(4)

    def fit(self, train_data, target):
        tensor_data_value = tr.from_numpy(train_data).float().reshape((train_data.shape[0], self.input_size))
        value_pred = self.value_model(tensor_data_value)

        value_target = target - np.abs(target / (target + 1E-20)) * value_pred.detach().numpy()
        tensor_data = tr.from_numpy(train_data).float().reshape((train_data.shape[0], 1, self.input_size))
        # tensor_target = tr.from_numpy(target)
        tensor_target = tr.from_numpy(value_target)
        output = self.model(tensor_data)

        self.model.zero_grad()
        output.backward(tensor_target.reshape((output.shape[0], output.shape[1], 1)))
        learning_rate = 0.03
        for f in self.model.parameters():
            f.data.add_(f.grad.data * learning_rate)

        value_loss_fn = nn.MSELoss()
        value_loss = value_loss_fn(value_pred, tr.from_numpy(np.sum(target, axis=1, keepdims=True)).float())
        self.value_model.zero_grad()
        value_loss.backward()
        learning_rate = 0.1

        for f in self.value_model.parameters():
            f.data.sub_(f.grad.data * learning_rate)

        for i in range(len(train_data)):
            result = self.predict(train_data[i:i+1, :])
            val_res = self.value_model(tensor_data[i:i+1, :])
            i = 0