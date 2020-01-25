import gym
import numpy as np

from network import Network

INPUT_SIZE = 29


def build_input(environment, state, prev_position):
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
    return env_vector, taxi_pos


def prepare_label(rewards):
    gamma = 0.9
    fake_labels = np.zeros((len(rewards), 6))
    for index, (act, rw) in enumerate(reversed(rewards)):
        value = np.zeros(6)
        value[act] = rw * (gamma ** index)
        fake_labels[index, :] = value
    return fake_labels


if __name__ == '__main__':
    network = Network(INPUT_SIZE)
    env = gym.make("Taxi-v3").env
    env.render()

    for i in range(1000000):
        new_state = env.s
        done = False
        k = 0
        frame, last_position = build_input(env, new_state, 0)
        samples = frame.reshape((1, INPUT_SIZE))
        rewards = list()
        while not done and k < 100:
            prev_act = env.lastaction

            action = network.predict(frame.reshape((1, INPUT_SIZE)))
            max_action = np.argmax(action)
            new_state, reward, done, info = env.step(max_action)

            rewards.append((max_action, reward))

            tmp_last_loc = last_position
            frame, last_position = build_input(env, new_state, last_position)
            samples = np.concatenate((samples, frame.reshape(1, INPUT_SIZE)), axis=0)

            k += 1
            if max_action != prev_act or last_position != tmp_last_loc:
                env.render()

        network.fit(samples[0:-1, :], prepare_label(rewards))
        env.reset()
