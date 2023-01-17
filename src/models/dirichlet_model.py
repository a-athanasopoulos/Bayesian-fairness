import copy
import numpy as np


class Model:
    """
    Probabilistic model
    """

    def __init__(self, Px, Py_x, Pz_yx):
        (self.Z, self.Y, self.X) = Pz_yx.shape

        self.Px = Px
        self.Py = np.zeros(shape=(self.Y, 1))
        self.Py_x = Py_x
        self.Px_y = np.zeros(shape=(self.X, self.Y))
        self.Px_yz = np.zeros(shape=(self.X, self.Y, self.Z))
        self.Pz_yx = Pz_yx
        self.Pxy = np.zeros(shape=(self.X, self.Y))
        self.Pyz = np.zeros(shape=(self.Y, self.Z))
        self.Pxyz = np.zeros(shape=(self.X, self.Y, self.Z))

        self.calculate_marginal_propabilities()

    def calculate_marginal_propabilities(self):
        # calculate Pxyz
        for x in range(self.X):
            for y in range(self.Y):
                for z in range(self.Z):
                    # calculate Pxyz
                    self.Pxyz[x, y, z] = self.Pz_yx[z, y, x] * self.Py_x[y, x] * self.Px[x]

        # calculate Px_yz
        for y in range(self.Y):
            for z in range(self.Z):
                self.Pyz[y, z] = np.sum(self.Pxyz[:, y, z])
                for x in range(self.X):
                    self.Pxy[x, y] = np.sum(self.Pxyz[x, y, :])
                    self.Px_yz[x, y, z] = self.Pxyz[x, y, z] / self.Pyz[y, z]

        # calculate Px_y
        for y in range(self.Y):
            self.Py[y] = np.sum(self.Pxyz[:, y, :])
            for x in range(self.X):
                self.Px_y[x, y] = self.Pxy[x, y] / self.Py[y]


class DirichletModel:
    def __init__(self, n_x, n_z, n_y, prior):
        """
        DirichletModel
        """
        self.X = n_x
        self.Y = n_y
        self.Z = n_z

        self.N_x = prior + np.zeros(shape=(n_x, 1))
        self.N_y_x = prior + np.zeros(shape=(n_y, n_x))
        self.N_z_yx = prior + np.zeros(shape=(n_z, n_y, n_x))

    def update_posterior_belief(self, data):
        """
        update posterior
        :param data:
        :return:
        """
        for i, datum in data.iterrows():
            self.N_x[datum["x"]] += 1
            self.N_y_x[datum["y"], datum["x"]] += 1
            self.N_z_yx[datum["z"], datum["y"], datum["x"]] += 1

    def get_marginal_model(self):
        """
        get marginal model
        :return:
        """
        Px = self.N_x / np.sum(self.N_x)
        Py_x = self.N_y_x / np.sum(self.N_y_x, axis=0)
        Pz_yx = self.N_z_yx / np.sum(self.N_z_yx, axis=0)
        return Model(Px=Px, Py_x=Py_x, Pz_yx=Pz_yx)

    def sample_model(self):
        """
        sample a model
        :return:
        """
        Px = np.zeros(shape=(self.X, 1))
        Py_x = np.zeros(shape=(self.Y, self.X))
        Pz_yx = np.zeros(shape=(self.Z, self.Y, self.X))

        Px = np.random.dirichlet(np.ravel(self.N_x)).reshape((-1, 1))

        for x in range(self.X):
            Py_x[:, x] = np.random.dirichlet(self.N_y_x[:, x])

        for x in range(self.X):
            for y in range(self.Y):
                Pz_yx[:, y, x] = np.random.dirichlet(self.N_z_yx[:, y, x])

        return Model(Px=Px, Py_x=Py_x, Pz_yx=Pz_yx)
