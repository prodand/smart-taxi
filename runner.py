from __future__ import absolute_import, division, print_function, unicode_literals

import sys
import time

import gym
import numpy as np

import tensorflow as tf
from tensorflow_core.python.keras.layers.core import Dense, Activation
from tensorflow_core.python.keras.models import Sequential
import tensorflow_core.python.keras.backend as K


def custom_loss(y_actual, y_predicted):
    return -y_actual * K.log(y_predicted)


INPUT_SIZE = 29
model = Sequential()
model.add(Dense(60, input_dim=INPUT_SIZE))
model.add(Activation('sigmoid'))
model.add(Dense(6))
model.add(Activation('softmax'))
model.compile(optimizer='adam',
              loss=custom_loss,
              metrics=['accuracy'])
env = gym.make("Taxi-v3").env


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


val = env.render(mode='ansi')

for i in range(1000000):
    new_state = env.s
    fake_labels = np.zeros((1, 6))
    done = False
    k = 0
    frame, last_position = build_input(env, new_state, 0)
    samples = frame.reshape((1, INPUT_SIZE))
    while not done and k < 100:
        prev_act = env.lastaction

        action = model.predict(frame.reshape((1, INPUT_SIZE)), batch_size=1)
        max_action = np.argmax(action)
        new_state, reward, done, info = env.step(max_action)

        fake_label = np.zeros((1, 6))
        fake_label[0, max_action] = reward
        fake_labels = np.concatenate((fake_labels, fake_label), axis=0)

        tmp_last_loc = last_position
        frame, last_position = build_input(env, new_state, last_position)
        samples = np.concatenate((samples, frame.reshape(1, INPUT_SIZE)), axis=0)

        k += 1
        if max_action != prev_act or last_position != tmp_last_loc:
            env.render()

    model.fit(samples[0:-1, :], fake_labels[1:, :])
    env.reset()
