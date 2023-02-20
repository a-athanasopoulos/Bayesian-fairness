import numpy as np


def get_delta(Px_y, Px_yz, **kwargs):
    """
    get model delta
    :return:
    """
    delta = np.zeros(Px_yz.shape)
    for z in range(Px_yz.shape[-1]):
        delta[:, :, z] = Px_y - Px_yz[:, :, z]
    return delta


def get_delta(Px_y, Px_yz, Pz_y):
    """
    get model delta
    :return:
    """
    n_x, n_y, n_z = Px_yz.shape
    delta = np.zeros((n_z, n_y, n_x))
    for z in range(Px_yz.shape[-1]):
        delta[z, :, :] = ((Px_y - Px_yz[:, :, z]) * Pz_y[z, :]).T
    return delta
