import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


def get_multivariate_normal_params(n_x):
    A = np.random.rand(n_x, n_x)
    cov = np.matmul(A.T, A)
    mean = np.random.rand(n_x)
    return mean, cov


def get_py_x(n_x, n_y):
    fake_samples = 1000
    fake_y = np.random.choice(range(n_y),
                              size=fake_samples)
    fake_x = np.random.normal(size=(fake_samples, n_x))

    model_y_x = LogisticRegression(n_jobs=-1)
    model_y_x.fit(fake_x, fake_y)

    init_weights = np.random.normal(0, 4, size=(1, n_x))
    model_y_x.coef_ = init_weights
    return model_y_x, init_weights


def get_pz_xy(n_x, n_y, n_z):
    fake_samples = 1000
    fake_y = np.random.choice(range(n_y), size=(fake_samples, 1))
    fake_z = np.random.choice(range(n_z), size=fake_samples)
    fake_x = np.random.normal(size=(fake_samples, n_x))

    input_data = np.concatenate([fake_y, fake_x], axis=1)
    model_z_yx = LogisticRegression(n_jobs=-1)
    model_z_yx.fit(input_data, fake_z)

    init_weights = np.random.normal(0, 0.5, size=(1, input_data.shape[1]))
    # model_z_yx.coef_ = init_weights
    return model_z_yx, init_weights


# all together
def generate_data(N, mean, cov, model_y_x, model_z_yx):
    x = np.random.multivariate_normal(mean, cov, N)
    y = model_y_x.predict(x)
    y = np.reshape(y, (-1, 1))
    z = model_z_yx.predict(np.concatenate([y, x], axis=1))
    z = np.reshape(z, (-1, 1))
    return x, y, z


def get_train_test(num_training, num_test, n_x, n_y, n_z):
    X_atr = [f"x_{i}" for i in range(n_x)]
    Y_atr = ["y"]
    Z_atr = "z"

    mean, cov = get_multivariate_normal_params(n_x)
    model_y_x, w = get_py_x(n_x, n_y)
    model_z_yx, w = get_pz_xy(n_x, n_y, n_z)
    # generate train set
    train_data = generate_data(num_training, mean, cov, model_y_x, model_z_yx)
    train_data = pd.DataFrame(np.concatenate(train_data, axis=1), columns=X_atr + Y_atr + [Z_atr]).astype("float32")

    test_data = generate_data(num_test, mean, cov, model_y_x, model_z_yx)
    test_data = pd.DataFrame(np.concatenate(test_data, axis=1), columns=X_atr + Y_atr + [Z_atr]).astype("float32")

    return train_data, test_data, X_atr, Y_atr, Z_atr


def get_synthetic_data(num_training, num_test):
    n_x, n_y, n_z = 6, 2, 12
    train_data, test_data, X_atr, Y_atr, Z_atr = get_train_test(num_training, num_test, n_x, n_y, n_z)

    return train_data, test_data, X_atr, Y_atr, Z_atr, n_y, n_z
