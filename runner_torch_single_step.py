import gym
import numpy as np
import time

from network import Network

INPUT_SIZE = 29


def build_input(environment, state, prev_position, ):
    # 25 for each position, 1 - for passenger location, 1 - for destination
    # 1 - last action, 1 - previous location
    env_vector = np.zeros(INPUT_SIZE, dtype=int)
    taxi_row, taxi_col, pass_loc, dest = environment.decode(state)
    taxi_pos = taxi_row * 5 + taxi_col
    env_vector[taxi_pos] = 1
    env_vector[25] = pass_loc + 1
    env_vector[26] = dest + 1
    env_vector[27] = 0 if env.lastaction is None else env.lastaction + 1
    env_vector[28] = prev_position
    env_vector[29] = prev_position
    env_vector[30] = prev_position
    return env_vector, taxi_pos


def prepare_labels(episode_rewards):
    discounted_rewards = np.zeros(len(episode_rewards), dtype=float)
    discounted_reward = 0
    gamma = 0.9
    for index, (act, rw) in enumerate(episode_rewards):
        discounted_reward = rw + discounted_reward * gamma
        discounted_rewards[index] = discounted_reward

    # discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-16)
    fake_labels = np.zeros((len(episode_rewards), 6))
    for index, (act, rw) in enumerate(episode_rewards):
        value = np.zeros(6, dtype=float)
        value[act] = discounted_rewards[index]
        fake_labels[index, :] = value
    return fake_labels


if __name__ == '__main__':
    network = Network(INPUT_SIZE)
    env = gym.make("Taxi-v3").env
    env.render()

    new_state = env.s
    done = False
    last_position = 0
    while not done:
        tmp_last_loc = last_position

        frame, last_position = build_input(env, new_state, last_position)
        rewards = list()
        prev_act = env.lastaction

        action = network.predict(frame.reshape((1, INPUT_SIZE)))
        max_action = np.argmax(action)
        new_state, reward, done, info = env.step(max_action)

        rewards.append((max_action, reward))

        if max_action != prev_act or last_position != tmp_last_loc:
            print()
            env.render()
            time.sleep(1)

        network.fit(frame.reshape(1, INPUT_SIZE), prepare_labels(rewards))
