import gym
import matplotlib.pyplot as plt
import numpy as np

from funcs import INPUT_SIZE, build_input, prepare_action_rewards
from monte.mc_network import McNetwork

env = gym.make("Taxi-v3").env
plt.ion()


def calculate_reward(old, new_s, steps):
    if old == new_s:
        return -1, True, False
    taxi_row, taxi_col, passenger, destination = env.decode(new_s)
    pass_loc = env.locs[passenger]
    if pass_loc[0] == taxi_row and pass_loc[1] == taxi_col:
        return 3, True, True
    if steps == 20:
        return -0.45, True, False
    return -0.07, False, False


def prepare_input(parking_states):
    result = build_input(env, parking_states[0]).reshape(INPUT_SIZE, 1)
    for i in range(1, len(parking_states)):
        result = np.column_stack((result, build_input(env, parking_states[i]).reshape(INPUT_SIZE, 1)))
    return result.T


def show_graphic(y_values, averages):
    # averageX = np.zeros((len(averages)))
    # for i, val in enumerate(averages):
    #     averageX[i] = i * 50
    plt.clf()
    # plt.plot(y_values)
    plt.plot(averages)
    plt.xlabel('Avg. %.4f' % averages[len(averages) - 1])
    plt.ylabel('Steps number')
    plt.show()


if __name__ == '__main__':
    network = McNetwork(INPUT_SIZE)
    env.reset()
    env.render()
    total_wins = 0
    iteration = 0
    performance = list()
    averages = list()
    avg = 0
    last_plot_point = 0
    episodes = 0
    while True:
        new_state = env.s
        k = 0
        frame = build_input(env, new_state)
        end = False

        print("----NEW ROUND----")
        env.render()
        my_reward = 0
        iteration += 1
        states = list()
        rewards = list()
        while not end:
            action = network.predict(frame)
            max_action = np.argmax(action)
            old_state = new_state
            new_state, reward, done, info = env.step(max_action)
            k += 1
            my_reward, end, reset = calculate_reward(old_state, new_state, k)

            print(action, my_reward, total_wins, "Episodes: ", episodes, "Iteration: ", iteration)
            print('Average: %.3f' % avg)
            env.render()

            states.append(old_state)
            rewards.append((max_action, my_reward))

            frame = build_input(env, new_state)
            if reset:
                env.reset()
            if end:
                break

        network.train(prepare_input(states), prepare_action_rewards(rewards))
        episodes += 1

        if my_reward > 1:
            total_wins += 1
            performance.append(episodes)
            episodes = 0

        if total_wins > last_plot_point and total_wins % 50 == 0:
            performance = performance[-1000:] if len(performance) > 1000 else performance
            averages.append(np.average(performance))
            avg = np.average(performance)
            averages = averages[-20:] if len(averages) > 20 else averages
            # show_graphic(performance, averages)
            last_plot_point = total_wins
