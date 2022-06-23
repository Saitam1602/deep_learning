import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_der(a):
    return a * (1 - a)


def tanh(z):
    return np.tanh(z)


def tanh_der(a):
    return 1 - np.power(a, 2)
