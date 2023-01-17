import numpy as np


def get_fairness(policy, model_delta):
    (X, Y, Z) = model_delta.shape
    fairness = 0
    for y in range(Y):
        for z in range(Z):
            delta = np.matmul(policy, model_delta[:, y, z])
            fairness += np.linalg.norm(delta, 1)
    return fairness


def get_fairness_gradient(policy, model_delta):
    fairness_gradient = np.zeros(policy.shape)

    (X, Y, Z) = model_delta.shape
    for y in range(Y):
        for z in range(Z):
            dyz = model_delta[:, y, z].reshape((-1, 1))
            c = np.matmul(policy, dyz)
            fairness_gradient -= np.matmul(c, dyz.T)

    return fairness_gradient
