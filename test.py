import numpy as np
import torch as tr
from torch import nn


def loss_simple(predicted, target):
    return tr.sum(-tr.log(predicted) * target, dim=1).mean()


model = nn.Sequential(
        nn.Linear(3, 3),
        nn.Softmax(dim=1))

data1 = np.array([1, 0, 1]).reshape((1, 3))
data2 = np.array([1, 1, 0]).reshape((1, 3))
for i in range(1000):

    actions1 = model(tr.from_numpy(data1).float()).detach().numpy()
    actions2 = model(tr.from_numpy(data2).float()).detach().numpy()

    tensor_data = tr.from_numpy(np.concatenate((data1, data2), axis=0)).float()
    res = model(tensor_data)

    model.zero_grad()

    res.backward(tr.tensor([
        [-10, 0, 0],
        [10, 0, 0]
    ], dtype=tr.float))
    # loss_fn = CrossEntropyLoss()
    # loss = loss_fn(res, tr.tensor([0, 0]))
    # loss = loss_simple(res, tr.from_numpy(np.array([
    #     [-1, 0, 0],
    #     [1, 0, 0]
    # ], dtype=tr.float)))

    # loss.backward()

    learning_rate = 0.01
    for f in model.parameters():
        f.data.add_(f.grad.data * learning_rate)
