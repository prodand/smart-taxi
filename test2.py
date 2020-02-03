import gym
import numpy as np

from dqn import build_fake_input
from dqn_network import DqnNetwork
from funcs import prepare_batch_inputs

INPUT_SIZE = 30


def print_values(pass_loc):
    field = np.zeros((5, 5), dtype=float)
    for idx in range(5):
        for j in range(5):
            env_vector = build_fake_input(idx, j, pass_loc)
            field[idx, j] = network.predict_value(env_vector)
    print(field)


network = DqnNetwork(INPUT_SIZE)

data = np.load("frames.npy")
rewards = np.load("rewards.npy")

# indexes = np.random.shuffle(np.arrange(0, data.shape[0]))
indexes = np.arange(0, data.shape[0])

for i, val in enumerate(indexes):
    inputs = data[i:i + 31, :].tolist()
    if i % 1000 == 0:
        print_values(0)
        print_values(1)
        print("===========")
    network.train_critic(prepare_batch_inputs(inputs), rewards[i:i + 31])

print_values(0)
print_values(1)
print_values(2)
print_values(3)

