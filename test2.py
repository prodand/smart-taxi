import numpy as np

from dqn_network import DqnNetwork

INPUT_SIZE = 30


def build_fake_input(taxi_row, taxi_col, passenger):
    env_vector = np.zeros(INPUT_SIZE, dtype=float)
    taxi_pos = taxi_row * 5 + taxi_col
    env_vector[taxi_pos] = 1
    env_vector[25] = 0
    env_vector[26] = 0
    env_vector[27] = 0
    env_vector[28] = 0
    env_vector[29] = 0
    env_vector[25 + passenger] = 0.5
    return env_vector


def print_values(pass_loc):
    field = np.zeros((5, 5), dtype=float)
    for idx in range(5):
        for j in range(5):
            env_vector = build_fake_input(idx, j, pass_loc)
            field[idx, j] = network.predict_value(env_vector)
    print(field)


def prepare_batch_inputs(frames):
    batch = np.zeros((len(frames), INPUT_SIZE))
    for fr_idx in range(len(frames)):
        batch[fr_idx, :] = frames[fr_idx]
    return batch


network = DqnNetwork(INPUT_SIZE)

data = np.load("frames-500.npy")
rewards = np.load("rewards-500.npy")

indexes = np.arange(0, data.shape[0])
prev_reward = 0
for i, val in enumerate(indexes):
    if prev_reward == 10.0:
        prev_reward = rewards[i]
        continue
    end_index = 1
    inputs = data[i:i + 2, :].tolist()
    reward = rewards[i]

    network.train_critic(prepare_batch_inputs(inputs), rewards[i:i + 1])

    print_values(0)
    print("--------------")
    prev_reward = reward

print_values(0)
print_values(1)
print_values(2)
print_values(3)
