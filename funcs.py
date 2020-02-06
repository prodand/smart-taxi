import numpy as np

INPUT_SIZE = 30


def build_input(environment, state):
    # 25 for each position, 5 - for passenger location, 4 - for destination
    env_vector = np.full(INPUT_SIZE, 0.0, dtype=float)
    cell_value = 0.01
    for i in range(25):
        env_vector[i] = cell_value
        cell_value += 0.01
    taxi_row, taxi_col, pass_loc, dest = environment.decode(state)
    taxi_pos = taxi_row * 5 + taxi_col
    env_vector[taxi_pos] = 1
    env_vector[25] = 0
    env_vector[26] = 0
    env_vector[27] = 0
    env_vector[28] = 0
    env_vector[29] = 0
    env_vector[25 + pass_loc] = 0.5
    return env_vector


def build_fake_input(taxi_row, taxi_col, passenger):
    env_vector = np.zeros(INPUT_SIZE, dtype=float)
    cell_value = 0.01
    for i in range(25):
        env_vector[i] = cell_value
        cell_value += 0.01
    taxi_pos = taxi_row * 5 + taxi_col
    env_vector[taxi_pos] = 1
    env_vector[25] = 0
    env_vector[26] = 0
    env_vector[27] = 0
    env_vector[28] = 0
    env_vector[29] = 0
    env_vector[25 + passenger] = 0.5
    return env_vector


def prepare_rewards(episode_rewards):
    size = len(episode_rewards)
    discounted_rewards = np.zeros(size, dtype=float)
    discounted_reward = 0
    gamma = 0.9
    for index, rw in enumerate(reversed(episode_rewards)):
        discounted_reward = rw  # + discounted_reward * gamma
        discounted_rewards[size - index - 1] = discounted_reward
    return discounted_rewards


def prepare_batch_inputs(frames):
    batch = np.zeros((len(frames) + 1, INPUT_SIZE))
    for i in range(len(frames)):
        batch[i, :] = frames[i]
    return batch
