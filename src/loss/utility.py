import numpy as np


def get_utility(policy, model, utility):
    """
    Calculate expected utility
    Todo: vectorize operation - minor
    """
    A, X = policy.shape
    Y = A
    Eu = 0
    for x in range(X):
        for y in range(Y):
            for a in range(A):
                Eu += utility[a, y] * policy[a, x] * model.Pxy[x, y]

    return Eu


def get_utility_gradient(model, utility):
    """
    Todo: vectorize operation
    """
    utility_gradient = np.matmul(utility, model.Pxy.T)
    return utility_gradient
