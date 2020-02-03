import numpy as np
import torch as tr
from torch import nn
from torch.nn import CrossEntropyLoss


class ConvNetwork:

    def __init__(self, input_size, env):
        self.input_size = input_size
        self.env = env
        self.model = nn.Sequential(
            nn.Conv1d(1, 6, input_size),
            nn.Sigmoid(),
            nn.Softmax(dim=1)
        )

    def predict(self, data):
        return self.model(tr.from_numpy(data).float().reshape((1, 1, self.input_size))).detach().numpy()

    def fit(self, train_data, target):
        tensor_data = tr.from_numpy(train_data).reshape((
            train_data.shape[0], 1, self.input_size
        )).float()
        # tensor_target = tr.from_numpy(np.sum(target, axis=1, keepdims=True)).long()
        tensor_target = tr.from_numpy(target)
        output = self.model(tensor_data)

        self.model.zero_grad()
        loss = self.loss_simple(output, tensor_target)
        # loss.backward()
        output.backward(loss.reshape(output.shape))

        learning_rate = 0.05
        for f in self.model.parameters():
            f.data.sub_(f.grad.data * learning_rate)

    def fit_single(self, train_data, target):
        tensor_data = tr.from_numpy(train_data).reshape((
            train_data.shape[0], 1, self.input_size
        )).float()
        tensor_target = tr.from_numpy(target)
        for i in range(len(tensor_data)):
            output = self.model(tensor_data[i:i + 1, :])

            self.model.zero_grad()
            # loss = self.loss_simple(output, tensor_target[i:i + 1, :])
            # loss.backward()
            output.backward(tensor_target[i:i + 1, :].reshape(1, 6, 1))

            learning_rate = 0.05
            for f in self.model.parameters():
                f.data.add_(f.grad.data * learning_rate)

    def loss_simple(self, predicted, target):
        # shape = predicted.shape
        # return tr.sum(tr.log(predicted.reshape((shape[0], shape[1]))) * target, dim=1).mean()
        shape = predicted.shape
        return tr.log(predicted.reshape((shape[0], shape[1]))) * target
