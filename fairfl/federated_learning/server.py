import warnings

warnings.filterwarnings('ignore')

import flwr as fl
import pandas as pd

import fl_utils
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from typing import Dict
from fairfl.metrics.bias_metrics import ACSIncomeBiasMetrics
from fairfl.metrics.classical_metrics import ClassicalMetrics
from client import client_fn

NUM_CLIENTS = 5


def fit_round(server_round: int) -> Dict:
    """Send round number to client."""
    return {"server_round": server_round}


def get_evaluate_fn(model: LogisticRegression):
    """Return an evaluation function for server-side evaluation."""

    # Load test data here to avoid the overhead of doing it in `evaluate` itself
    x_test, y_test = fl_utils.load_test_dataset()

    # The `evaluate` function will be called after every round
    def evaluate(server_round, parameters: fl.common.NDArrays, config):
        fl_utils.set_model_params(model, parameters)
        loss = log_loss(y_test, model.predict_proba(x_test.values))
        predicted = model.predict(x_test.values)
        test_df = pd.concat(
            [x_test.reset_index(drop=True), y_test.to_frame(),
             pd.DataFrame(predicted, columns=["PINCP_predicted"])],
            axis=1)
        accuracy, precision, recall, f1 = ClassicalMetrics.calculate_classical_metrics(y_pred=predicted, y_true=y_test)
        di, sp, eo, eod = ACSIncomeBiasMetrics("gender").return_bias_metrics(test_df)
        return loss, {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1,
                      "disparate impact": di,
                      "statistical parity": sp, "equal opportunity": eo, "equal opportunity diff": eod}

    return evaluate


# Start Flower server for five rounds of federated learning
if __name__ == "__main__":
    model = LogisticRegression()
    fl_utils.set_initial_params(model)
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=0.5,
        min_fit_clients=NUM_CLIENTS,
        min_evaluate_clients=3,
        min_available_clients=NUM_CLIENTS,
        evaluate_fn=get_evaluate_fn(model),
        on_fit_config_fn=fit_round,
    )

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=5),
        strategy=strategy
    )
