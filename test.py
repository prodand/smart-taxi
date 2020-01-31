import numpy as np
import torch as tr
from torch import nn


def gradient(predicted, q_value):
    shape = predicted.shape
    return -tr.log(predicted).reshape((shape[0], shape[1])) * q_value # + tr.tensor([[1, 0, 0]])


def create_hot_encoded_vector(advantage, action):
    vector = tr.zeros(3)
    vector[action] = advantage.max()
    return vector


model = nn.Sequential(
    nn.Linear(3, 3),
    nn.Softmax(dim=1))

data1 = np.array([1, 0, 1]).reshape((1, 3))
# data2 = np.array([1, 1, 0]).reshape((1, 3))
model[0].weight.data = tr.nn.Parameter(tr.tensor([[0.2588, 0.4934, 0.1695],
                                                  [-0.1624, -0.3345, -0.4680],
                                                  [0.2181, -0.0168, 0.1223]]))
model[0].bias.data = tr.nn.Parameter(tr.tensor([-0.3653,  0.3581, -0.3080]))

for i in range(1000):

    # actions1 = model(tr.from_numpy(data1).float()).detach().numpy()
    # actions2 = model(tr.from_numpy(data2).float()).detach().numpy()

    tensor_data = tr.from_numpy(data1).float()
    res = model(tensor_data)

    loss = gradient(res, create_hot_encoded_vector(tr.tensor([[-0.336]]), 0))
    model.zero_grad()
    res.backward(loss)

    learning_rate = 0.1
    for f in model.parameters():
        f.data.add_(f.grad.data * learning_rate)

# res.backward(tr.tensor([
#     [-10, 0, 0],
#     [10, 0, 0]
# ], dtype=tr.float))

# loss_fn = CrossEntropyLoss()
# loss = loss_fn(res, tr.tensor([0, 0]))
# loss = loss_simple(res, tr.from_numpy(np.array([
#     [-1, 0, 0],
#     [1, 0, 0]
# ], dtype=tr.float)))

# loss.backward()
