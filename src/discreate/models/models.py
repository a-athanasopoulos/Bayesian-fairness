import numpy as np
import pandas as pd


def get_models(data, Z_atr, X_atr, Y_atr, n_y, n_z):
    # add a fake sample with missing z values to make sklearn model produce every propability
    unique_z_values = data[Z_atr].unique()
    missing_z = [z for z in range(n_z) if z not in unique_z_values]
    if missing_z:
        fake_df = data.sample(len(missing_z), replace=True)
        fake_df[Z_atr] = missing_z
        data = pd.concat([data, fake_df], ignore_index=True)
        # print(missing_z)

    # Py, Pz_y
    N_y = np.zeros(n_y) + 0.5  # np.finfo(np.float32).eps
    N_z_y = np.zeros((n_z, n_y)) + 0.5  # np.finfo(np.float32).eps
    for i, datum in data.iterrows():
        N_y[int(datum[Y_atr])] += 1
        N_z_y[int(datum[Z_atr]), int(datum[Y_atr])] += 1
    Py = N_y / np.sum(N_y)
    Pz_y = N_z_y / np.sum(N_z_y, axis=0)

    # Py_x
    from sklearn.linear_model import LogisticRegression
    model_y_x = LogisticRegression(max_iter=2000, n_jobs=-1)
    model_y_x.fit(X=data[X_atr], y=data[Y_atr[0]])

    # Pz_yx
    input_features = Y_atr + X_atr
    model_z_yx = LogisticRegression(max_iter=2000, n_jobs=-1)
    model_z_yx.fit(X=data[input_features], y=data[Z_atr])

    return Py, Pz_y, model_y_x, model_z_yx


def get_models_from_data(data, X_atr, Y_atr, Z_atr, n_y, n_z):
    # get Py, Pz_y and models for Py_x & Pz_yx
    Py, Pz_y, model_y_x, model_z_yx = get_models(data=data,
                                                 X_atr=X_atr,
                                                 Y_atr=Y_atr,
                                                 Z_atr=Z_atr,
                                                 n_y=n_y,
                                                 n_z=n_z)

    # get Py_x from data
    Py_x = model_y_x.predict_proba(data[X_atr]).T

    # get Pz_yx from data
    Pz_yx = np.zeros((n_z, n_y, data.shape[0]))
    for y in range(n_y):
        tmp_data = data[Y_atr + X_atr].copy()
        tmp_data[Y_atr] = y
        Pz_yx[:, y, :] = model_z_yx.predict_proba(tmp_data[Y_atr + X_atr]).T
    return Py, Pz_y, Py_x, Pz_yx


def get_data_from_models(data, Py, Pz_y, model_y_x, model_z_yx, X_atr, Y_atr, n_z, n_y):
    Py_x = model_y_x.predict_proba(data[X_atr]).T

    # get Pz_yx from data
    Pz_yx = np.zeros((n_z, n_y, data.shape[0]))
    for y in range(n_y):
        tmp_data = data[Y_atr + X_atr].copy()
        tmp_data[Y_atr] = y
        Pz_yx[:, y, :] = model_z_yx.predict_proba(tmp_data[Y_atr + X_atr]).T
    return Py, Pz_y, Py_x, Pz_yx


def get_bootstrap_models(dataset, bootstrap_models, X_atr, Y_atr, Z_atr, n_y, n_z):
    bootstrap_data_models = []
    bootstrap_datasets = []
    for n in range(bootstrap_models):
        bootstrap_dataset = dataset.sample(frac=1.0, replace=True).reset_index(drop=True)
        bootstrap_datasets += [bootstrap_dataset]
        bootstrap_data_models += [get_models(data=bootstrap_dataset,
                                             X_atr=X_atr,
                                             Y_atr=Y_atr,
                                             Z_atr=Z_atr,
                                             n_y=n_y,
                                             n_z=n_z)]
    return bootstrap_data_models


def get_data_from_boostrap_model(data, all_data, bootstrap_models, X_atr, Y_atr, Z_atr, n_y, n_z):
    bootstrap_data_models = []
    for n in range(bootstrap_models):
        bootstrap_dataset = all_data.sample(frac=1.0, replace=True).reset_index(drop=True)
        Py, Pz_y, model_y_x, model_z_yx = get_models(data=bootstrap_dataset,
                                                     X_atr=X_atr,
                                                     Y_atr=Y_atr,
                                                     Z_atr=Z_atr,
                                                     n_y=n_y,
                                                     n_z=n_z)

        bootstrap_data_models += [get_data_from_models(data=data,
                                                       Py=Py,
                                                       Pz_y=Pz_y,
                                                       model_y_x=model_y_x,
                                                       model_z_yx=model_z_yx,
                                                       X_atr=X_atr, Y_atr=Y_atr,
                                                       n_z=n_z, n_y=n_y)]
    return bootstrap_data_models


def get_data_from_model(data, all_data, X_atr, Y_atr, Z_atr, n_y, n_z):
    models = get_models(data=all_data,
                        X_atr=X_atr, Y_atr=Y_atr, Z_atr=Z_atr,
                        n_y=n_y, n_z=n_z)
    Py, Pz_y, model_y_x, model_z_yx = models
    Py, Pz_y, Py_x, Pz_yx = get_data_from_models(data=data,
                                                 Py=Py,
                                                 Pz_y=Pz_y,
                                                 model_y_x=model_y_x,
                                                 model_z_yx=model_z_yx,
                                                 X_atr=X_atr, Y_atr=Y_atr,
                                                 n_z=n_z, n_y=n_y)
    return Py, Pz_y, Py_x, Pz_yx
