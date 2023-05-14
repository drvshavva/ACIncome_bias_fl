import warnings

warnings.filterwarnings('ignore')

import flwr as fl
import pandas as pd

import fl_utils
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from typing import Dict
from metrics.bias_metrics import ACSIncomeBiasMetrics
from client_virtual import client_fn

NUM_CLIENTS = 10


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
        loss = log_loss(y_test, model.predict_proba(x_test))
        predicted = model.predict(x_test)
        test_df = pd.concat(
            [x_test.reset_index(drop=True), y_test.to_frame(),
             pd.DataFrame(predicted, columns=["PINCP_predicted"])],
            axis=1)
        accuracy = model.score(x_test, y_test)
        di, sp, eo, eod = ACSIncomeBiasMetrics().return_bias_metrics(test_df)
        return loss, {"accuracy": accuracy, "disparate impact": di,
                      "statistical parity": sp, "equal opportunity": eo, "equal opportunity diff": eod}

    return evaluate


# Start Flower server for five rounds of federated learning
if __name__ == "__main__":
    model = LogisticRegression()
    fl_utils.set_initial_params(model)
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,  # Sample 100% of available clients for training
        fraction_evaluate=1.0,  # Sample 100% of available clients for evaluation
        min_fit_clients=NUM_CLIENTS,  # Never sample less than 50 clients for training
        min_evaluate_clients=NUM_CLIENTS,  # Never sample less than 50 clients for evaluation
        min_available_clients=NUM_CLIENTS,  # Wait until all 50 clients are available
        evaluate_fn=get_evaluate_fn(model),
        on_fit_config_fn=fit_round,
    )
    # fl.server.start_server(
    #     server_address="127.0.0.1:8080",
    #     strategy=strategy,
    #     config=fl.server.ServerConfig(num_rounds=5),
    # )
    # Start simulation
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=5),
        strategy=strategy
    )
