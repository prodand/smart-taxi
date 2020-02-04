import numpy as np

from dqn import build_fake_input
from dqn_network import DqnNetwork

INPUT_SIZE = 30


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
        for uuu in range(25):
            frames[fr_idx][uuu] = 0.2 if frames[fr_idx][uuu] == 0.0 else frames[fr_idx][uuu]
        batch[fr_idx, :] = frames[fr_idx]
    return batch


network = DqnNetwork(INPUT_SIZE)

data = np.load("frames.npy")
rewards = np.load("rewards.npy")

# indexes = np.random.shuffle(np.arrange(0, data.shape[0]))
indexes = np.arange(0, data.shape[0])

for i, val in enumerate(indexes):
    end_index = 1
    inputs = data[i:i + 2, :].tolist()
    reward = rewards[i]
    network.train_critic(prepare_batch_inputs(inputs), rewards[i:i + 1])
    print_values(0)
    print("--------------")

print_values(0)
print_values(1)
print_values(2)
print_values(3)
