import torch as tr
from torch import nn

ACTION_SIZE = 4


class Flatten(nn.Module):
    def forward(self, x):
        x = x.view(x.size()[0], -1)
        return x


class DqnNetwork:

    def __init__(self, input_size):
        self.input_size = input_size
        self.model = nn.Sequential(
            nn.Conv1d(1, ACTION_SIZE, input_size),
            nn.Sigmoid(),
            # Flatten(),
            # nn.Linear(4, 4),
            # nn.Sigmoid()
            nn.Softmax(dim=1)
        )
        self.critic_model = nn.Sequential(
            nn.Conv1d(1, 4, input_size),
            nn.Sigmoid(),
            Flatten(),
            nn.Linear(4, 1)
        )

    def predict(self, data):
        return self.model(tr.from_numpy(data).float().reshape((1, 1, self.input_size))) \
            .detach().numpy().reshape(ACTION_SIZE)

    def train(self, old_state, new_state, reward, action):
        tensor_data = self.create_tensor(old_state)

        q_value_old = self.critic_model(tensor_data.clone().detach())
        q_value_new = self.critic_model(self.create_tensor(new_state)) if reward != -1 else 0
        advantage = (reward + 0.9 * q_value_new) - q_value_old
        self.critic_model.zero_grad()
        q_value_old.backward(advantage.clone().detach())
        for f in self.critic_model.parameters():
            f.data.add_(f.grad.data * 0.01)

        output = self.model(tensor_data)
        loss = self.gradient(output, self.create_hot_encoded_vector(advantage.clone().detach(), action))
        self.model.zero_grad()
        loss.backward()
        for f in self.model.parameters():
            f.data.add_(f.grad.data * 0.01)

        output = self.model(tensor_data)

    def create_tensor(self, data):
        return tr.from_numpy(data).float().reshape((1, 1, self.input_size))

    def create_hot_encoded_vector(self, advantage, action):
        vector = tr.zeros(ACTION_SIZE)
        vector[action] = advantage.max()
        return vector

    def create_entropy(self, predicted):
        return tr.sum(tr.log(predicted.max(1)[0]) * predicted.max(1)[0])

    def gradient(self, predicted, q_value):
        shape = predicted.shape
        entropy = self.create_entropy(predicted.clone().detach())
        return tr.sum(tr.log(predicted).reshape((shape[0], shape[1])) * q_value) + entropy
