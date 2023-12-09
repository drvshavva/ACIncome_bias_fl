import warnings

import flwr as fl
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from fairfl.metrics.bias_metrics import ACSIncomeBiasMetrics, BiasType
from fairfl.metrics.classical_metrics import ClassicalMetrics

import fl_utils

seed = 10


class ACSIncomeClient(fl.client.NumPyClient):
    def __init__(self, model, x_train, y_train, state, x_test, y_test, bias_type):
        self.model = model
        self.x_train = x_train
        self.y_train = y_train
        self.state = state
        self.x_test = x_test
        self.y_test = y_test
        self.bias_type = bias_type
        self.num_p = 188234
        self.num_up = 4063

    def get_parameters(self, config):  # type: ignore
        return fl_utils.get_model_parameters(self.model)

    def calculate_f_global_segment_and_local_metric(self):
        predicted = self.model.predict(self.x_test)
        test_df = pd.concat(
            [self.x_test.reset_index(drop=True), self.y_test.to_frame(),
             pd.DataFrame(predicted, columns=["PINCP_predicted"])],
            axis=1)
        _, sp_local, _ = ACSIncomeBiasMetrics(self.bias_type).return_bias_metrics(test_df)

        # şimdi f_global için segmenti bu client için hesaplayalım
        priviliged = test_df[test_df["RAC1P"] == 1]  # others
        unpriviliged = test_df[test_df["RAC1P"] == 0]  # black
        predictions_unp = unpriviliged[unpriviliged['PINCP_predicted'] == 1]
        predictions_p = priviliged[priviliged['PINCP_predicted'] == 1]

        segment = ((predictions_unp.shape[0] * unpriviliged.shape[0] / self.num_up) -
                   (predictions_p.shape[0] * priviliged.shape[0] / self.num_p))
        return segment, sp_local

    def fit(self, parameters, config):  # type: ignore
        fl_utils.set_model_params(self.model, parameters)
        # Ignore convergence failure due to low local epochs
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(self.x_train, self.y_train)
        segment, sp_local = self.calculate_f_global_segment_and_local_metric()
        print(f"Training finished for round {config['server_round']}")
        # print(f"Number of training samples in state: {self.state} is {len(self.x_train)}")
        return fl_utils.get_model_parameters(self.model), len(self.x_train), {f"segment": segment,
                                                                              "len_test": len(self.x_test),
                                                                              "sp": sp_local,
                                                                              "state": self.state}

    def evaluate(self, parameters, config):  # type: ignore
        fl_utils.set_model_params(self.model, parameters)
        loss = log_loss(self.y_test, self.model.predict_proba(self.x_test))
        predicted = self.model.predict(self.x_test.values)
        test_df = pd.concat(
            [self.x_test.reset_index(drop=True), self.y_test.to_frame(),
             pd.DataFrame(predicted, columns=["PINCP_predicted"])],
            axis=1)
        accuracy, precision, recall, f1 = ClassicalMetrics.calculate_classical_metrics(y_pred=predicted,
                                                                                       y_true=self.y_test)
        di, sp, eod = ACSIncomeBiasMetrics(self.bias_type).return_bias_metrics(test_df)
        # print(f"Number of test samples in state: {self.state} is {len(self.x_test)}")
        return loss, len(self.x_test), {"f1": f1,
                                        "disparate impact": di,
                                        "statistical parity": sp,
                                        "equal opportunity diff": eod}


def client_fn(cid: str) -> ACSIncomeClient:
    """Create a client representing a single state."""
    # from fairfl.utils.constants import states
    states = ["MT", "WY", "ID", "VT", "UT", "SD", "ME", "NH", "OR", "ND", "IA", "AK", "HI", "NM", "NE", "CO", "WI",
              "MN",
              "WA", "WV"]

    bias_type = BiasType.race

    # Create LogisticRegression Model
    model = LogisticRegression(random_state=seed, solver="liblinear", C=0.5)
    # Setting initial parameters, akin to model.compile for keras models
    fl_utils.set_initial_params(model)

    # get dataset for selected state
    state = states[int(cid)]
    x_train, y_train, x_test, y_test = fl_utils.load_data_for_state(state)

    partition_id = np.random.choice(5)
    (x_train, y_train) = fl_utils.partition(x_train.values, y_train.values, 5)[partition_id]

    # Create a  single client representing a single state
    return ACSIncomeClient(model=model,
                           x_train=x_train,
                           y_train=y_train,
                           state=state,
                           x_test=x_test,
                           y_test=y_test,
                           bias_type=bias_type)
