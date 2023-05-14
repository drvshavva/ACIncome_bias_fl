import numpy as np
from sklearn.linear_model import LogisticRegression

from utils.read_data import DatasetUtils
from model.preprocess import ACSIncomePreprocess

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
    n_features = 568  # Number of features in dataset
    model.classes_ = np.array([i for i in range(10)])

    model.coef_ = np.zeros((n_classes, n_features))
    if model.fit_intercept:
        model.intercept_ = np.zeros((n_classes,))


def load_data_for_state(state):
    train, test = preprocess.get_preprocessed_data()
    train_state, test_state = utils.get_state_data(train, test, state)
    x_train, y_train, x_test, y_test = preprocess.get_x_y_preprocessed(train_state, test_state)
    return x_train, y_train, x_test, y_test


def load_test_dataset():
    _, test = preprocess.get_preprocessed_data()
    test = test.drop('ST', axis=1)
    return preprocess.split_x_y(test)


def shuffle(X: np.ndarray, y: np.ndarray):
    """Shuffle X and y."""
    rng = np.random.default_rng()
    idx = rng.permutation(len(X))
    return X[idx], y[idx]


def partition(X: np.ndarray, y: np.ndarray, num_partitions: int):
    """Split X and y into a number of partitions."""
    return list(
        zip(np.array_split(X, num_partitions), np.array_split(y, num_partitions))
    )
