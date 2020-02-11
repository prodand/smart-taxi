import torch as tr
from torch import nn
from torch.nn import L1Loss
from torch.optim import SGD

ACTION_SIZE = 4


class Flatten(nn.Module):
    def forward(self, x):
        x = x.view(x.size()[0], -1)
        return x


class DqnBaselineNetwork:

    def __init__(self, input_size):
        self.input_size = input_size
        self.input_shape = (1, self.input_size)
        self.model = nn.Sequential(
            nn.Linear(input_size, 16),
            nn.Sigmoid(),
            nn.Linear(16, 8),
            nn.Tanh(),
            nn.Linear(8, 4),
        )
        self.critic_model = nn.Sequential(
            nn.Linear(input_size, 16),
            nn.ReLU(),
            nn.Linear(16, 4),
            nn.Sigmoid(),
            nn.Linear(4, 1)
        )

    def predict_value(self, data):
        return self.critic_model(self.create_tensor(data)).detach().numpy().reshape(1)

    def predict(self, data):
        output = self.model(self.create_tensor(data))
        return nn.functional.softmax(output, 1).detach().numpy().reshape(ACTION_SIZE)

    def train(self, states, action_rewards):
        for i in range(len(states)):
            old_state = states[i]
            action, reward = action_rewards[i]
            q_value = self.critic_model(self.create_tensor(old_state))

            self.model.zero_grad()
            output = self.model(self.create_tensor(old_state))
            advantage = q_value.clone().detach().max() if reward == 0 else tr.tensor(reward)
            loss = self.gradient(output, self.create_hot_encoded_vector(tr.tensor(1.0), action))
            output.backward(advantage * loss.detach())
            for f in self.model.parameters():
                f.data.add_(f.grad.data * 0.1)
            output_next = self.predict(old_state)
            i = 1

    def train_critic(self, states, targets):
        for i in range(len(states) - 2, -1, -1):
            old_state = states[i]
            new_state = states[i + 1]
            reward = targets[i]
            self.update_critic(old_state, new_state, reward)

    def update_critic(self, old_state, new_state, reward):
        optimizer = SGD(self.critic_model.parameters(), lr=0.07)
        loss_fn = L1Loss()
        q_value_new = self.critic_model(self.create_tensor(new_state)).detach().numpy() \
            if reward != -1 and reward != 10.0 else 0

        self.critic_model.zero_grad()
        q_value_old = self.critic_model(self.create_tensor(old_state))
        tensor_target = tr.tensor([reward + 0.6 * q_value_new]).reshape(q_value_old.shape)
        loss = loss_fn(q_value_old, tensor_target)
        loss.backward(loss)
        optimizer.step()
        return q_value_old

    def gradient(self, predicted, expected):
        # entropy = self.create_entropy(predicted.clone().detach())
        return expected - nn.functional.softmax(predicted, dim=1)

    def create_hot_encoded_vector(self, advantage, action):
        vector = tr.zeros(1, ACTION_SIZE)
        action = int(action)
        vector[0, action] = advantage.max()
        return vector

    def create_tensor(self, data):
        return tr.from_numpy(data).float().reshape(self.input_shape)

    def create_batch_tensor(self, data):
        return tr.from_numpy(data).float().reshape((32, self.input_size))

    def create_entropy(self, predicted):
        action = predicted.argmax()
        return tr.sum(tr.log(predicted) * predicted) * self.create_hot_encoded_vector(tr.tensor(1), action)
