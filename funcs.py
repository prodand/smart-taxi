import numpy as np

INPUT_SIZE = 25


def cell_to_position(cell):
    return cell[0] * 5 + cell[1]


def build_input(environment, state):
    env_vector = np.full(INPUT_SIZE, 0.0, dtype=float)
    taxi_row, taxi_col, pass_loc, dest = environment.decode(state)
    taxi_pos = cell_to_position((taxi_row, taxi_col))
    if pass_loc != 4:
        passenger_cell = environment.locs[pass_loc]
        pass_pos = cell_to_position(passenger_cell)
    else:
        pass_pos = taxi_pos
    destination_cell = environment.locs[dest]
    dest_pos = cell_to_position(destination_cell)
    env_vector[cell_to_position(environment.locs[0])] = 0.1
    env_vector[cell_to_position(environment.locs[1])] = 0.1
    env_vector[cell_to_position(environment.locs[2])] = 0.1
    env_vector[cell_to_position(environment.locs[3])] = 0.1
    env_vector[taxi_pos] = 1
    env_vector[dest_pos] += 0.7
    env_vector[pass_pos] += 0.4
    return env_vector


def build_fake_input(taxi_row, taxi_col, passenger):
    env_vector = np.zeros(INPUT_SIZE, dtype=float)
    taxi_pos = taxi_row * 5 + taxi_col
    pickups = [(0, 0), (0, 4), (4, 0), (4, 3)]
    passenger_cell = pickups[passenger]
    pass_pos = passenger_cell[0] * 5 + passenger_cell[1]
    env_vector[pass_pos] = 0.5
    env_vector[taxi_pos] = 1
    return env_vector


def prepare_rewards(rewards):
    size = len(rewards)
    discounted_rewards = np.zeros(size, dtype=float)
    discounted_reward = 0
    gamma = 0.9
    for index, rw in enumerate(reversed(rewards)):
        discounted_reward = rw  # + discounted_reward * gamma
        discounted_rewards[size - index - 1] = discounted_reward
    return discounted_rewards


def prepare_action_rewards(action_rewards):
    size = len(action_rewards)
    discounted_rewards = np.zeros((size, 2), dtype=float)
    discounted_reward = 0
    gamma = 0.9
    for index, (action, rw) in enumerate(reversed(action_rewards)):
        discounted_reward = rw + discounted_reward * gamma
        discounted_rewards[size - index - 1] = [action, discounted_reward]
    return discounted_rewards


def prepare_batch_inputs(frames):
    batch = np.zeros((len(frames) + 1, INPUT_SIZE))
    for i in range(len(frames)):
        batch[i, :] = frames[i]
    return batch
