import numpy as np
from sklearn.linear_model import LogisticRegression

from fairfl.utils.read_data import DatasetUtils
from fairfl.model.preprocess import ACSIncomePreprocess
from aif360.algorithms.preprocessing import Reweighing
from aif360.datasets import BinaryLabelDataset

utils = DatasetUtils()
preprocess = ACSIncomePreprocess()


def get_model_parameters(model: LogisticRegression):
    """Returns the paramters of a sklearn LogisticRegression model."""
    if model.fit_intercept:
        params = [
            model.coef_,
            model.intercept_,
        ]
    else:
        params = [
            model.coef_,
        ]
    return params


def set_model_params(model: LogisticRegression, params) -> LogisticRegression:
    """Sets the parameters of a sklean LogisticRegression model."""
    model.coef_ = params[0]
    if model.fit_intercept:
        model.intercept_ = params[1]
    return model


def set_initial_params(model: LogisticRegression):
    """Sets initial parameters as zeros Required since model params are
    uninitialized until model.fit is called.

    But server asks for initial parameters from clients at launch. Refer
    to sklearn.linear_model.LogisticRegression documentation for more
    information.
    """
    n_classes = 2
    # n_features = 575  # Number of features in dataset
    n_features = 567  # Number of features in dataset
    model.classes_ = np.array([i for i in range(2)])

    model.coef_ = np.zeros((n_classes, n_features))
    if model.fit_intercept:
        model.intercept_ = np.zeros((n_classes,))


def apply_reweighing_race(df):
    train_dataset = BinaryLabelDataset(df=df,
                                       favorable_label=1, unfavorable_label=0,
                                       label_names=['PINCP'], protected_attribute_names=['RAC1P'],
                                       privileged_protected_attributes=[[1]],
                                       unprivileged_protected_attributes=[[0]])
    privileged_groups = [{'RAC1P': 1}]
    unprivileged_groups = [{'RAC1P': 0}]

    RW = Reweighing(unprivileged_groups=unprivileged_groups,
                    privileged_groups=privileged_groups)
    dataset_transf_train = RW.fit_transform(train_dataset)
    return dataset_transf_train.convert_to_dataframe()[0]


def load_data_for_state(state):
    train, test = preprocess.get_preprocessed_data()
    train_state, test_state = utils.get_state_data(train, test, state)
    del train
    del test
    # tüm denemelerde kullanılan test verisi kullanıldı
    # test = test.drop('ST', axis=1)
    train_state_processed = apply_reweighing_race(train_state)
    x_train, y_train, x_test, y_test = preprocess.get_x_y_preprocessed(train_state_processed, test_state)
    del train_state
    del test_state
    return x_train, y_train, x_test, y_test


def load_test_dataset(states):
    _, test = preprocess.get_preprocessed_data()
    test_states = utils.get_selected_states(test, states)
    # test = test.drop('ST', axis=1)
    return preprocess.split_x_y(test_states)


def partition(X: np.ndarray, y: np.ndarray, num_partitions: int):
    """Split X and y into a number of partitions."""
    return list(
        zip(np.array_split(X, num_partitions), np.array_split(y, num_partitions))
    )


def shuffle(X: np.ndarray, y: np.ndarray):
    """Shuffle X and y."""
    rng = np.random.default_rng()
    idx = rng.permutation(len(X))
    return X[idx], y[idx]
