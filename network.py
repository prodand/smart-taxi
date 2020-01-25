import torch as tr
from torch import nn


class Network:

    def __init__(self, input_size):
        self.model = nn.Sequential(
            nn.Linear(input_size, 60),
            nn.Sigmoid(),
            nn.Linear(60, 6),
            nn.Softmax()
        )

    def predict(self, data):
        return self.model(data)

    def fit(self, train_data, target):
        output = self.model(train_data)

        self.model.zero_grad()
        loss = self.loss_simple(output, target)
        loss.backward()

        learning_rate = 0.01
        for f in self.model.parameters():
            f.data.sub_(f.grad.data * learning_rate)

    def loss_simple(self, predicted, target):
        return -tr.log(predicted) * target
