import numpy as np


class CrossEntropy:

    def loss(self, probs, expected):
        return -np.log(np.sum(np.multiply(probs, expected)) + 1e-20)

    def delta(self, probs: np.array, expected: np.array) -> np.array:
        return probs - expected
