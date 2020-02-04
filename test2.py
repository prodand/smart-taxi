import numpy as np

from dqn_network import DqnNetwork
from layers.custom_net import CustomNet
from layers.fully_connected import FullyConnected

INPUT_SIZE = 30


def build_fake_input(taxi_row, taxi_col, passenger):
    env_vector = np.zeros(INPUT_SIZE, dtype=float)
    tmp_val = 0.02
    for uuu in range(25):
        env_vector[uuu] = tmp_val if env_vector[uuu] == 0.0 else env_vector[uuu]
        tmp_val += 0.01
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


def print_custom_values(pass_loc):
    field = np.zeros((5, 5), dtype=float)
    for idx in range(5):
        for j in range(5):
            env_vector = build_fake_input(idx, j, pass_loc)
            field[idx, j] = custom.predict(env_vector)
    print(field)


def prepare_batch_inputs(frames):
    batch = np.zeros((len(frames), INPUT_SIZE))
    tmp_val = 0.02
    for fr_idx in range(len(frames)):
        for uuu in range(25):
            frames[fr_idx][uuu] = tmp_val if frames[fr_idx][uuu] == 0.0 else frames[fr_idx][uuu]
            tmp_val += 0.01
        batch[fr_idx, :] = frames[fr_idx]
    return batch


network = DqnNetwork(INPUT_SIZE)

data = np.load("frames.npy")
rewards = np.load("rewards.npy")

custom = CustomNet(1, 0.1)
# custom.add_layer(FullyConnected(INPUT_SIZE, 8,
#                                 weights=network.critic_model[0].weight.data.numpy().T,
#                                 bias=network.critic_model[0].bias.data.numpy().T,
#                                 sigm=True))
# custom.add_layer(FullyConnected(8, 4,
#                                 weights=network.critic_model[2].weight.data.numpy().T,
#                                 bias=network.critic_model[2].bias.data.numpy().T,
#                                 sigm=True))
custom.add_layer(FullyConnected(INPUT_SIZE, 1,
                                weights=network.critic_model[0].weight.data.numpy().T,
                                bias=network.critic_model[0].bias.data.numpy().T,
                                ))

# indexes = np.random.shuffle(np.arrange(0, data.shape[0]))
indexes = np.arange(0, data.shape[0])

for i, val in enumerate(indexes):
    end_index = 1
    inputs = data[i:i + 2, :].tolist()
    reward = rewards[i]
    network.train_critic(prepare_batch_inputs(inputs), rewards[i:i + 1])
    custom.learn(prepare_batch_inputs(inputs).reshape((2, INPUT_SIZE)), rewards[i:i + 1])
    print_values(0)
    print_custom_values(0)
    print("--------------")

print_values(0)
print_values(1)
print_values(2)
print_values(3)
