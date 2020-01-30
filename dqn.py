import gym
import numpy as np

from dqn_network import DqnNetwork
from runner_torch import build_input

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


def calculate_reward(old_state, new_state, steps):
    if old_state == new_state:
        return -1, True
    taxi_row, taxi_col, passenger, destination = env.decode(new_state)
    pass_loc = env.locs[passenger]
    if pass_loc[0] == taxi_row and pass_loc[1] == taxi_col:
        return 1, True
    if steps == 20:
        return -1, True
    return 0, False


if __name__ == '__main__':
    network = DqnNetwork(INPUT_SIZE)

    while True:
        env.reset()
        new_state = env.s
        k = 0
        frame, tmp = build_input(env, new_state)
        samples = frame.reshape((1, INPUT_SIZE))
        rewards = list()
        end = False
        print("----NEW ROUND----")
        env.render()
        while not end:
            action = network.predict(frame)
            max_action = np.argmax(action)
            old_state = new_state
            new_state, reward, done, info = env.step(max_action)
            k += 1
            my_reward, end = calculate_reward(old_state, new_state, k)
            print(action, my_reward)
            env.render()

            rewards.append((max_action, my_reward))

            frame, tmp = build_input(env, new_state)
            samples = np.concatenate((samples, frame.reshape(1, INPUT_SIZE)), axis=0)

        env.render()
        network.fit(samples[0:-1, :], prepare_labels(rewards))
