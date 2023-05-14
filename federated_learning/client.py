import warnings
import sys

import flwr as fl
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from metrics.bias_metrics import ACSIncomeBiasMetrics

import fl_utils


def start_client_for_selected_state(state):
    x_train, y_train, x_test, y_test = fl_utils.load_data_for_state(state)

    # Split train set into 10 partitions and randomly use one for training.
    partition_id = np.random.choice(10)
    (X_train, y_train) = fl_utils.partition(x_train.values, y_train.values, 10)[partition_id]

    # Create LogisticRegression Model
    model = LogisticRegression()

    # Setting initial parameters, akin to model.compile for keras models
    fl_utils.set_initial_params(model)

    class ACSIncomeClient(fl.client.NumPyClient):
        def get_parameters(self, config):  # type: ignore
            return fl_utils.get_model_parameters(model)

        def fit(self, parameters, config):  # type: ignore
            fl_utils.set_model_params(model, parameters)
            # Ignore convergence failure due to low local epochs
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_train, y_train)
            print(f"Training finished for round {config['server_round']}")
            return fl_utils.get_model_parameters(model), len(X_train), {}

        def evaluate(self, parameters, config):  # type: ignore
            fl_utils.set_model_params(model, parameters)
            loss = log_loss(y_test, model.predict_proba(x_test))
            predicted = model.predict(x_test)
            test_df = pd.concat(
                [x_test.reset_index(drop=True), y_test.to_frame(),
                 pd.DataFrame(predicted, columns=["PINCP_predicted"])],
                axis=1)
            accuracy = model.score(x_test, y_test)
            di, sp, eo, eod = ACSIncomeBiasMetrics().return_bias_metrics(test_df)
            return loss, len(x_test), {"accuracy": accuracy, "disparate impact": di,
                                       "statistical parity": sp, "equal opportunity": eo, "equal opportunity diff": eod}

    # Start Flower client
    fl.client.start_numpy_client(server_address="10.8.131.175:8080", client=ACSIncomeClient())


if __name__ == '__main__':
    from utils.constants import states

    start_client_for_selected_state(states[int(sys.argv[1])])
