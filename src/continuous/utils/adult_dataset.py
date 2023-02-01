import pandas as pd
import numpy as np
from collections import OrderedDict

from src.continuous.utils.data_utils import encode_data

data_types = OrderedDict([
    ("age", "int"),
    ("workclass", "category"),
    ("final_weight", "int"),  # originally it was called fnlwgt
    ("education", "category"),
    ("education_num", "int"),
    ("marital_status", "category"),
    ("occupation", "category"),
    ("relationship", "category"),
    ("race", "category"),
    ("sex", "category"),
    ("capital_gain", "float"),  # required because of NaN values
    ("capital_loss", "int"),
    ("hours_per_week", "int"),
    ("native_country", "category"),
    ("income_class", "category"),
])
target_column = "income_class"


def read_dataset(path):
    return pd.read_csv(
        path,
        names=data_types,
        index_col=None,

        comment='|',  # test dataset has comment in it
        skipinitialspace=True,  # Skip spaces after delimiter
        na_values={
            'capital_gain': 99999,
            'workclass': '?',
            'native_country': '?',
            'occupation': '?',
        },
        dtype=data_types,
    )


def clean_dataset(data):
    # Test dataset has dot at the end, we remove it in order
    # to unify names between training and test datasets.
    data['income_class'] = data.income_class.str.rstrip('.').astype('category')

    # Remove final weight column since there is no use
    # for it during the classification.
    data = data.drop('final_weight', axis=1)

    # Duplicates might create biases during the analysis and
    # during prediction stage they might give over-optimistic
    # (or pessimistic) results.
    data = data.drop_duplicates()

    # Binary target variable (>50K == 1 and <=50K == 0)
    data[target_column] = (data[target_column] == '>50K').astype(int)

    # Categorical dataset
    categorical_features = data.select_dtypes('category').columns
    data[categorical_features] = data.select_dtypes('category').apply(lambda x: x.cat.codes)
    return data


def get_adult_datasets(data_path):
    # org features
    X_atr = ['age', 'workclass', 'education', 'education_num', 'marital_status',
             'occupation', 'relationship', 'capital_gain',
             'capital_loss', 'hours_per_week', 'native_country']
    Y_atr = ["income_class"]
    Z_atr = ['race', 'sex']

    # load train
    data_path = data_path + "/adult"
    train_data = clean_dataset(read_dataset(data_path + "/adult.data"))
    train_data = train_data.dropna()

    # load test
    test_data = clean_dataset(read_dataset(data_path + "/adult.test"))
    test_data = test_data.dropna()

    dataset = pd.concat([train_data, test_data])

    # preprossesing
    unique_z = np.unique(dataset[Z_atr].values, axis=0)
    n_z = len(unique_z)

    unique_y = np.unique(dataset[Y_atr].values, axis=0)
    n_y = len(unique_y)

    train_data["z"] = encode_data(train_data[Z_atr], unique_values=unique_z)
    train_data.drop(Z_atr, axis=1, inplace=True)

    test_data["z"] = encode_data(test_data[Z_atr], unique_values=unique_z)
    test_data.drop(Z_atr, axis=1, inplace=True)
    Z_atr = "z"

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    train_data[X_atr] = scaler.fit_transform(train_data[X_atr])
    test_data[X_atr] = scaler.transform(test_data[X_atr])

    all_features = X_atr + Y_atr + [Z_atr]
    train_data = train_data.reset_index(drop=True).astype("float32")
    test_data = test_data.reset_index(drop=True).astype("float32")
    return train_data[all_features], test_data[all_features], X_atr, Y_atr, Z_atr, n_y, n_z


if __name__ == "__main__":
    base_path = "/Users/andreasathanasopoulos/Phd/projects/bayesian_fairness/my_code/Bayesian-fairness/data"
    train_data, test_data, X_atr, Y_atr, Z_atr, n_y, n_z = get_adult_datasets(base_path)

    print("testttt")
