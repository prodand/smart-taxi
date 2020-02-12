import gym
import numpy as np
import matplotlib.pyplot as plt

from dqn_baseline_network import DqnBaselineNetwork
from funcs import INPUT_SIZE, build_input, prepare_batch_inputs, prepare_rewards, build_fake_input

env = gym.make("Taxi-v3").env

def calculate_reward(old, new_s, steps):
    if old == new_s:
        return -1, True
    taxi_row, taxi_col, passenger, destination = env.decode(new_s)
    pass_loc = env.locs[passenger]
    if pass_loc[0] == taxi_row and pass_loc[1] == taxi_col:
        return 10, True
    if steps == 20:
        return -0.8, True
    return 0, False


def print_values(state):
    taxi_row, taxi_col, pass_loc, dest = env.decode(state)
    field = np.zeros((5, 5), dtype=float)
    for i in range(5):
        for j in range(5):
            env_vector = build_fake_input(i, j, pass_loc)
            field[i, j] = network.predict_value(env_vector)
    print(field)


def prepare_input(current_state, next_state):
    return np.column_stack((build_input(env, current_state), build_input(env, next_state))).T


def show_graphic(y_values):
    plt.plot(y_values)
    plt.ylabel('Steps number')
    plt.show()


if __name__ == '__main__':
    network = DqnBaselineNetwork(INPUT_SIZE)
    env.reset()
    new_state = env.s
    env.render()
    total_wins = 0
    steps_to_win = 0
    print_values(env.s)
    iteration = 0
    performance = list()
    last_plot_point = 0
    while True:
        k = 0
        frame = build_input(env, new_state)
        end = False
        print("----NEW ROUND----")
        my_reward = 0
        states = list()
        rewards = list()
        iteration += 1
        while not end:
            action = network.predict(frame)
            max_action = np.argmax(action)
            old_state = new_state
            new_state, reward, done, info = env.step(max_action)
            k += 1
            steps_to_win += 1
            my_reward, end = calculate_reward(old_state, new_state, k)

            print(action, my_reward, total_wins, "Steps: ", steps_to_win, "Iteration: ", iteration)
            env.render()

            states.append(build_input(env, old_state))
            rewards.append(my_reward)
            if end:
                network.train_critic(prepare_batch_inputs(states), prepare_rewards(rewards))

            print_values(env.s)

            frame = build_input(env, new_state)
            network.train(prepare_input(old_state, new_state), [(max_action, my_reward)])
            aaaa = 0

        if my_reward == 10:
            total_wins += 1
            performance.append(steps_to_win)
            steps_to_win = 0
            new_state = env.s
            env.reset()

        if total_wins > last_plot_point and total_wins % 50 == 0:
            performance = performance[0:-1000] if len(performance) > 1000 else performance
            show_graphic(performance)
            last_plot_point = total_wins
