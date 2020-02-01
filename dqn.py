import gym
import numpy as np

from dqn_network import DqnNetwork

INPUT_SIZE = 34
env = gym.make("Taxi-v3").env


def prepare_labels(episode_rewards):
    discounted_rewards = np.zeros(len(episode_rewards), dtype=float)
    discounted_reward = 0
    gamma = 0.9
    for index, (act, rw) in enumerate(reversed(episode_rewards)):
        discounted_reward = rw + discounted_reward * gamma
        discounted_rewards[len(episode_rewards) - index - 1] = discounted_reward

    fake_labels = np.zeros((len(episode_rewards), 4))
    for index, (act, rw) in enumerate(episode_rewards):
        value = np.zeros(4, dtype=float)
        value[act] = discounted_rewards[index]
        fake_labels[index, :] = value
    return fake_labels


def build_input(environment, state):
    # 25 for each position, 5 - for passenger location, 4 - for destination
    env_vector = np.zeros(INPUT_SIZE, dtype=int)
    taxi_row, taxi_col, pass_loc, dest = environment.decode(state)
    taxi_pos = taxi_row * 5 + taxi_col
    env_vector[taxi_pos] = 1
    env_vector[25 + pass_loc] = 5
    env_vector[30 + dest] = 20
    return env_vector


def calculate_reward(old_state, new_state, steps):
    if old_state == new_state:
        return -1, False
    taxi_row, taxi_col, passenger, destination = env.decode(new_state)
    pass_loc = env.locs[passenger]
    if pass_loc[0] == taxi_row and pass_loc[1] == taxi_col:
        return 10, True
    # if steps == 20:
    #     return -1, True
    return 0, False


if __name__ == '__main__':
    network = DqnNetwork(INPUT_SIZE)

    env.reset()
    new_state = env.s
    env.render()
    while True:
        k = 0
        frame = build_input(env, new_state)
        end = False
        print("----NEW ROUND----")
        my_reward = 0
        while not end:
            action = network.predict(frame)
            max_action = np.argmax(action)
            old_state = new_state
            new_state, reward, done, info = env.step(max_action)
            k += 1
            my_reward, end = calculate_reward(old_state, new_state, k)

            print(action, my_reward)
            env.render()

            frame = build_input(env, new_state)
            network.train(build_input(env, old_state), frame, my_reward, max_action)
        if end:
            new_state = env.s
            env.reset()
