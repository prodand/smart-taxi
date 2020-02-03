import time

import gym
import numpy as np
import torch as tr
from torch import nn

from dqn import build_fake_input
from dqn_network import DqnNetwork
from runner_torch import build_input, prepare_labels

INPUT_SIZE = 30

network = DqnNetwork(INPUT_SIZE)

for i in range(10000000):
    print("Iteration: ", i)
    res = network.predict_value(build_fake_input(1, 1, 2))
    print(res)
    res = network.predict_value(build_fake_input(2, 0, 2))
    print(res)
    res = network.predict_value(build_fake_input(0, 4, 2))
    print(res)
    network.train_critic(build_fake_input(1, 1, 2), [[-1]])
    network.train_critic(build_fake_input(2, 0, 2), [[1]])