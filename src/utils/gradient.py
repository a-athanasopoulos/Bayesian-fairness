import numpy as np


def project_gradient(grad):
    proj_grad = grad - np.mean(grad, axis=0)
    return proj_grad
