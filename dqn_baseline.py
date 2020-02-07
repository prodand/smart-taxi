import gym
import numpy as np

from dqn_baseline_network import DqnBaselineNetwork
from funcs import INPUT_SIZE, build_input, prepare_batch_inputs, prepare_rewards, build_fake_input
from layers.custom_net_actor import CustomNetActor
from layers.fully_connected import FullyConnected
from layers.softmax import Softmax

env = gym.make("Taxi-v3").env


def calculate_reward(old, new_s, steps):
    if old == new_s:
        return -1, True
    taxi_row, taxi_col, passenger, destination = env.decode(new_s)
    pass_loc = env.locs[passenger]
    if pass_loc[0] == taxi_row and pass_loc[1] == taxi_col:
        return 1, True
    if steps == 20:
        return -0.2, True
    return 0, False


def print_values(state):
    taxi_row, taxi_col, pass_loc, dest = env.decode(state)
    field = np.zeros((5, 5), dtype=float)
    for i in range(5):
        for j in range(5):
            env_vector = build_fake_input(i, j, pass_loc)
            field[i, j] = network.predict_value(env_vector)
    print(field)


def prepare_input(current_state):
    return build_input(env, current_state).reshape((1, INPUT_SIZE))


if __name__ == '__main__':
    custom = CustomNetActor(1, 0.1)
    network = DqnBaselineNetwork(INPUT_SIZE)
    custom.add_layer(FullyConnected(INPUT_SIZE, 16,
                                    weights=network.model[0].weight.data.numpy().T,
                                    bias=network.model[0].bias.data.numpy().T,
                                    ))
    custom.add_layer(FullyConnected(16, 8,
                                    weights=network.model[2].weight.data.numpy().T,
                                    bias=network.model[2].bias.data.numpy().T,
                                    ))
    custom.add_layer(FullyConnected(8, 4,
                                    weights=network.model[4].weight.data.numpy().T,
                                    bias=network.model[4].bias.data.numpy().T,
                                    ))
    custom.add_layer(Softmax())

    env.reset()
    new_state = env.s
    env.render()
    total_wins = 0
    steps_to_win = 0
    print_values(env.s)
    iteration = 0
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
            action1 = custom.predict(frame)
            max_action = np.argmax(action)
            old_state = new_state
            new_state, reward, done, info = env.step(max_action)
            k += 1
            steps_to_win += 1
            my_reward, end = calculate_reward(old_state, new_state, k)

            print(action, my_reward, total_wins, "Steps: ", steps_to_win, "Iteration: ", iteration)
            print(action1)
            env.render()

            print_values(env.s)

            states.append(build_input(env, old_state))
            rewards.append(my_reward)
            if end:
                network.train_critic(prepare_batch_inputs(states), prepare_rewards(rewards))

            network.train(prepare_input(old_state), [(max_action, my_reward)])
            custom.learn(prepare_input(old_state), [(max_action, my_reward)])
            frame = build_input(env, new_state)

        if my_reward == 10:
            total_wins += 1
            steps_to_win = 0
            new_state = env.s
            env.reset()
