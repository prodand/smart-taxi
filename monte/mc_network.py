import numpy as np
import torch as tr
from torch import nn

ACTION_SIZE = 6
GAMMA = 0.6
ADVANTAGE_THRESHOLD = 0.015


def to_arrays(action_rewards):
    actions = np.zeros(len(action_rewards))
    rewards = np.zeros(len(action_rewards), dtype=float)
    for i in range(len(action_rewards)):
        action, reward = action_rewards[i]
        actions[i] = action
        rewards[i] = reward
    return actions, rewards


class McNetwork:

    def __init__(self, input_size):
        self.input_size = input_size
        self.model = nn.Sequential(
            nn.Linear(input_size, 16),
            nn.Sigmoid(),
            nn.Linear(16, ACTION_SIZE),
        )

    def predict(self, data):
        output = self.model(self.create_tensor(data, 1))
        return nn.functional.softmax(output, 1).detach().numpy().reshape(ACTION_SIZE)

    def train(self, states, action_rewards):
        actions, rewards = to_arrays(action_rewards)
        indexes = list(range(0, len(states)))
        np.random.shuffle(indexes)
        for i in indexes:
            state = states[i:i+1]
            action = actions[i:i+1]
            before = self.predict(state)
            self.model.zero_grad()
            output = self.model(self.create_tensor(state, len(state)))
            loss = self.gradient(output, self.create_hot_encoded_vector(action))
            # output.backward((loss.detach().T * tr.tensor(rewards[i:i+1])).T.detach())
            G = tr.tensor(rewards[i:i+1]).max()
            loss.backward()
            for f in self.model.parameters():
                f.data.add_(G * f.grad.data * 0.007)
            after = self.predict(state)
            idx = int(action.max())
            if G < 0 and before[idx] < after[idx]:
                print("Wrong negative")
            if G > 0 and before[idx] > after[idx]:
                print("Wrong positive")

    def gradient(self, predicted, expected):
        return tr.sum(expected * nn.functional.log_softmax(predicted, dim=1))

    def create_hot_encoded_vector(self, actions):
        vector = np.zeros((len(actions), ACTION_SIZE))
        vector[:, actions.astype(int)] = 1
        return tr.tensor(vector).float()

    def create_tensor(self, data, batch_size):
        return tr.from_numpy(data).float().reshape(batch_size, self.input_size)
