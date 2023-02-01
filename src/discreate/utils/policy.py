import numpy as np


def get_random_policy(size):
    a = np.ones(shape=size[0])
    policy = np.random.dirichlet(a, size=size[1])
    return np.transpose(policy)


def get_random_policy_2(size):
    policy = np.random.random(size=size)
    return normalize_policy(policy)


def normalize_policy(policy):
    policy[policy < 0] = 0
    policy[policy > 1] = 1
    for x in range(policy.shape[-1]):
        policy[:, x] /= np.sum(policy[:, x])
    return policy
