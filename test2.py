import numpy as np

from dqn import build_fake_input
from dqn_network import DqnNetwork
from layers.custom_net import CustomNet
from layers.fully_connected import FullyConnected

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

data = np.load("frames-500.npy")
rewards = np.load("rewards-500.npy")

custom = CustomNet(1, 0.1)
custom.add_layer(FullyConnected(INPUT_SIZE, 8, sigm=True))
custom.add_layer(FullyConnected(8, 4, sigm=True))
custom.add_layer(FullyConnected(4, 1))

# indexes = np.random.shuffle(np.arrange(0, data.shape[0]))
indexes = np.arange(0, data.shape[0])

for i, val in enumerate(indexes):
    end_index = 1
    inputs = data[i:i + 2, :].tolist()
    reward = rewards[i]
    if reward != 0:
        network.train_critic(prepare_batch_inputs(inputs), rewards[i:i + 1])
        custom.learn(prepare_batch_inputs(inputs).reshape((2, INPUT_SIZE)), rewards[i:i + 1])
        print_values(0)
        print("--------------")

print_values(0)
print_values(1)
print_values(2)
print_values(3)
