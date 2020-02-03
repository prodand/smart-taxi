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
        self.input_shape = (1, self.input_size)
        self.model = nn.Sequential(
            nn.Linear(input_size, 8),
            nn.Sigmoid(),
            nn.Linear(8, 4),
            nn.Softmax(dim=1)
        )
        self.critic_model = nn.Sequential(
            nn.Linear(input_size, 8),
            nn.Sigmoid(),
            nn.Linear(8, 4),
            nn.Sigmoid(),
            nn.Linear(4, 1)
        )

    def predict_value(self, data):
        return self.critic_model(self.create_tensor(data)).detach().numpy().reshape(1)

    def predict(self, data):
        return self.model(self.create_tensor(data)).detach().numpy().reshape(ACTION_SIZE)

    def train(self, states, action_rewards):
        for i in range(len(states) - 1):
            old_state = states[i]
            new_state = states[i + 1]
            action, reward = action_rewards[i]
            q_value_old = self.critic_model(self.create_tensor(old_state))
            q_value_new = self.critic_model(self.create_tensor(new_state)) if reward != -1 else 0
            advantage = (reward + 0.9 * q_value_new) - q_value_old
            self.critic_model.zero_grad()
            q_value_old.backward(advantage.clone().detach())
            for f in self.critic_model.parameters():
                f.data.add_(f.grad.data * 0.05)

            output = self.model(self.create_tensor(old_state))
            loss = self.gradient(output, self.create_hot_encoded_vector(q_value_old.clone().detach(), action))
            self.model.zero_grad()
            output.backward(loss)
            for f in self.model.parameters():
                f.data.add_(f.grad.data * 0.1)

            output = self.model(self.create_tensor(old_state))

    def create_tensor(self, data):
        return tr.from_numpy(data).float().reshape(self.input_shape)

    def create_batch_tensor(self, data):
        return tr.from_numpy(data).float().reshape((32, self.input_size))

    def create_hot_encoded_vector(self, advantage, action):
        vector = tr.zeros(1, ACTION_SIZE)
        action = int(action)
        vector[0, action] = advantage.max()
        return vector

    def create_entropy(self, predicted):
        action = predicted.argmax()
        return tr.sum(tr.log(predicted) * predicted) * self.create_hot_encoded_vector(tr.tensor(1), action)

    def gradient(self, predicted, q_value):
        entropy = self.create_entropy(predicted.clone().detach())
        return -tr.log(predicted) * q_value + 0.001 * entropy

    def train_critic(self, states, targets):
        self.critic_model.zero_grad()
        q_value_old = self.critic_model(self.create_batch_tensor(states))
        errors = tr.zeros(q_value_old.shape)
        for i in range(len(states) - 1):
            old_state = states[i]
            reward = targets[i]
            q_value_old = self.critic_model(self.create_tensor(old_state))
            q_value_new = self.critic_model(self.create_tensor(states[i + 1])) if reward != -1 else 0
            advantage = (reward + 0.9 * q_value_new) - q_value_old
            q_value_old.backward(advantage.clone().detach())
        for f in self.critic_model.parameters():
            f.data.add_(f.grad.data * 0.01)