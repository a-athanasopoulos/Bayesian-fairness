import numpy as np
import pandas as pd

from src.discrete.models.dirichlet_model import MarginalModel


class BoostrapModel(object):
    def __init__(self, n_x, n_z, n_y):
        self.X = n_x
        self.Y = n_y
        self.Z = n_z

        self.N_x = np.zeros(shape=(n_x, 1)) + 0.5
        self.N_y_x = np.zeros(shape=(n_y, n_x)) + 0.5
        self.N_z_yx = np.zeros(shape=(n_z, n_y, n_x)) + 0.5

    def sample_model(self, data: pd.DataFrame):
        sample = data.sample(n=data.shape[0], replace=True)

        for i, datum in sample.iterrows():
            self.N_x[datum["x"]] += 1
            self.N_y_x[datum["y"], datum["x"]] += 1
            self.N_z_yx[datum["z"], datum["y"], datum["x"]] += 1

        Px = self.N_x / np.sum(self.N_x)
        Py_x = self.N_y_x / np.sum(self.N_y_x, axis=0)
        Pz_yx = self.N_z_yx / np.sum(self.N_z_yx, axis=0)
        return MarginalModel(Px=Px, Py_x=Py_x, Pz_yx=Pz_yx)
