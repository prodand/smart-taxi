import torch as tr
from torch import nn


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
        tensor_data = tr.from_numpy(train_data).float()
        output = self.model(tensor_data)

        self.model.zero_grad()
        loss = self.loss_simple(output, tr.from_numpy(target))
        loss.backward()

        learning_rate = 0.01
        for f in self.model.parameters():
            f.data.sub_(f.grad.data * learning_rate)

    def fit_single(self, train_data, target, state):
        tensor_data = tr.from_numpy(train_data).reshape((
            train_data.shape[0], 1, self.input_size
        )).float()
        tensor_target = tr.from_numpy(target)
        for i in range(len(tensor_data)):
            output = self.model(tensor_data[i:i + 1, :])

            self.model.zero_grad()
            loss = self.distance(state) - self.loss_simple(output, tensor_target[i:i + 1, :])
            loss.backward()

            learning_rate = 0.05
            for f in self.model.parameters():
                f.data.add_(f.grad.data * learning_rate)

    def loss_simple(self, predicted, target):
        return tr.sum(-tr.log(predicted).reshape(6) * target.reshape(6))

    def distance(self, state):
        taxi_row, taxi_col, passenger, dest = self.env.decode(state)
        dest_loc = tr.tensor(self.env.locs[dest])
        pass_loc = tr.tensor([taxi_row, taxi_col] if passenger == 4 else self.env.locs[passenger])
        taxi_loc = tr.tensor([taxi_row, taxi_col])
        distance = tr.abs(taxi_loc - pass_loc).sum() + tr.abs(pass_loc - dest_loc).sum()
        return distance
