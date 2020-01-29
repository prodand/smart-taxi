import time

import gym
import numpy as np

from conv_network import ConvNetwork

INPUT_SIZE = 34


def build_input(environment, state):
    # 25 for each position, 5 - for passenger location, 4 - for destination
    env_vector = np.zeros(INPUT_SIZE, dtype=int)
    taxi_row, taxi_col, pass_loc, dest = environment.decode(state)
    taxi_pos = taxi_row * 5 + taxi_col
    env_vector[taxi_pos] = 1
    env_vector[25 + pass_loc] = 5
    env_vector[30 + dest] = 20
    return env_vector, taxi_pos


def prepare_labels(episode_rewards):
    discounted_rewards = np.zeros(len(episode_rewards), dtype=float)
    discounted_reward = 0
    gamma = 0.9
    for index, (act, rw) in enumerate(reversed(episode_rewards)):
        discounted_reward = rw + discounted_reward * gamma
        discounted_rewards[len(episode_rewards) - index - 1] = discounted_reward

    # discounted_rewards = discounted_rewards / 10
    # center = (discounted_rewards.max() + discounted_rewards.min()) / 2
    # discounted_rewards = (discounted_rewards - center)
    # discounted_rewards = (discounted_rewards - center) / (discounted_rewards.std() + 1e-16)
    fake_labels = np.zeros((len(episode_rewards), 6))
    for index, (act, rw) in enumerate(episode_rewards):
        value = np.zeros(6, dtype=float)
        value[act] = discounted_rewards[index]
        fake_labels[index, :] = value
    return fake_labels


if __name__ == '__main__':
    env = gym.make("Taxi-v3").env
    env.render()
    network = ConvNetwork(INPUT_SIZE, env)

    while True:
        new_state = env.s
        k = 0
        frame, last_position = build_input(env, new_state)
        samples = frame.reshape((1, INPUT_SIZE))
        rewards = list()
        done = False
        while not done and k < 1:
            prev_act = env.lastaction

            action = network.predict(frame.reshape((1, INPUT_SIZE)))
            max_action = np.argmax(action)
            new_state, reward, done, info = env.step(max_action)
            rewards.append((max_action, reward))

            tmp_last_loc = last_position
            frame, last_position = build_input(env, new_state)
            samples = np.concatenate((samples, frame.reshape(1, INPUT_SIZE)), axis=0)

            k += 1
            if max_action != prev_act or last_position != tmp_last_loc or k > 8:
                print(action.reshape(6), reward)
                env.render()
                time.sleep(1)
            if done:
                env.reset()

            network.fit_single(samples[0:-1, :], prepare_labels(rewards), new_state)