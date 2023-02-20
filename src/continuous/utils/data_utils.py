import pandas as pd
import numpy as np


def load_compas_dataset(path):
    """
    load COMPAS dataset and clip feature according to the clip_value
    :return: pd.Dataframe
    """
    raw_data = pd.read_csv(path + "/compas.csv")
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


def get_continuous_compas_dataset(data_path, shuffle=0, norm=True):
    Z_atr = ["sex", "race"]
    X_atr = ['age_cat', 'juv_fel_count', 'juv_misd_count', 'juv_other_count', 'priors_count', 'c_charge_degree']
    Y_atr = ['two_year_recid']

    # load dataset
    data = load_compas_dataset(path=data_path)

    # get distinct values
    unique_z = np.unique(data[Z_atr].values, axis=0)
    n_z = len(unique_z)

    unique_y = np.unique(data[Y_atr].values, axis=0)
    n_y = len(unique_y)

    data["z"] = encode_data(data[Z_atr], unique_values=unique_z)
    Z_atr = "z"

    data = data.astype("float32")

    from sklearn.preprocessing import StandardScaler
    if norm:
        data[X_atr] = StandardScaler().fit_transform(data[X_atr])

    if shuffle:
        data = data.sample(frac=1.0, replace=False)
    train_data = data.iloc[0:6000]
    test_data = data.iloc[6000:]

    return train_data, test_data, X_atr, Y_atr, Z_atr, n_y, n_z
