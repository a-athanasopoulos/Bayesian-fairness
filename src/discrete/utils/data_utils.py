import pandas as pd
import numpy as np


def load_compas_dataset(path, clip_features, clip_value):
    """
    load COMPAS dataset and clip feature according to the clip_value
    :return: pd.Dataframe
    """
    raw_data = pd.read_csv(path + "/compas.csv")
    raw_data[raw_data[clip_features] > clip_value] = clip_value
    return raw_data


def encode_data(data, unique_values):
    """
    covert dataset to discrete version
    :return: pd.DataFrame
    """
    encoded_value = np.array([])
    for i, d in data.iterrows():
        # encode feature to an index represents the unique value.
        index = np.argmax((d.values == unique_values).all(axis=1))
        encoded_value = np.append(encoded_value, index)
    return encoded_value.astype(int)


def create_compact_dataset(data, Z_atr, X_atr, Y_atr, unique_x, unique_y, unique_z):
    # encode z
    encoded_z = encode_data(data[Z_atr], unique_z)
    assert (int(max(encoded_z)) + 1) == len(unique_z)

    # encode x
    encoded_x = encode_data(data[X_atr], unique_x)
    assert (int(max(encoded_x)) + 1) == len(unique_x)

    # encode y
    encoded_y = encode_data(data[Y_atr], unique_y)
    assert (int(max(encoded_y)) + 1) == len(unique_y)

    return np.stack([encoded_x, encoded_z, encoded_y], axis=1)


def get_discrete_compas_dataset(data_path):
    Z_atr = ["sex", "race"]
    X_atr = ['age_cat', 'juv_fel_count', 'juv_misd_count', 'juv_other_count', 'priors_count', 'c_charge_degree']
    Y_atr = ['two_year_recid']

    # features to clip
    clip_features = ["juv_fel_count", "juv_misd_count", "juv_other_count", "priors_count"]
    clip_value = 2

    # load dataset
    data = load_compas_dataset(path=data_path, clip_features=clip_features, clip_value=clip_value)

    # get distinct values
    unique_z = np.unique(data[Z_atr].values, axis=0)
    n_z = len(unique_z)

    unique_x = np.unique(data[X_atr].values, axis=0)
    n_x = len(unique_x)

    unique_y = np.unique(data[Y_atr].values, axis=0)
    n_y = len(unique_y)

    dataset = create_compact_dataset(data, Z_atr, X_atr, Y_atr, unique_x, unique_y, unique_z)
    dataset = pd.DataFrame(dataset, columns=["x", "z", "y"])

    train_data = dataset.iloc[0:6000]
    test_data = dataset.iloc[6000:]
    return train_data, test_data, X_atr, Y_atr, Z_atr, n_x, n_y, n_z
