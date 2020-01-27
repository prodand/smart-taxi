import torch as tr
from torch import nn


class Network:

    def __init__(self, input_size):
        # self.model = nn.Sequential(
        #     nn.Linear(input_size, 60),
        #     nn.Sigmoid(),
        #     nn.Linear(60, 6),
        #     nn.Softmax(dim=1)
        # )
        self.model = nn.Sequential(
            nn.Linear(input_size, 5),
            nn.Sigmoid(),
            nn.Linear(5, 6),
            nn.Softmax(dim=1)
        )

    def predict(self, data):
        return self.model(tr.from_numpy(data).float()).detach().numpy()

    def fit(self, train_data, target):
        tensor_data = tr.from_numpy(train_data).float()
        output = self.model(tensor_data)

        self.model.zero_grad()
        # loss = self.loss_simple(output, tr.from_numpy(target))
        # loss.backward()
        output.backward(tr.from_numpy(target))

        learning_rate = 0.01
        for f in self.model.parameters():
            f.data.add_(f.grad.data * learning_rate)

    def loss_simple(self, predicted, target):
        return tr.sum(-tr.log(predicted) * target, dim=1).mean()
