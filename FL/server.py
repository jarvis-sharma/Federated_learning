import argparse
from typing import Callable, Dict, Optional, Tuple,List

import numpy as np

import flwr as fl

import modelFunctionsServer
from sklearn.metrics import classification_report

def main() -> None:
    DEFAULT_SERVER_ADDRESS = "[::]:8080"
    DATASET_PATH1='./X_train3.npy'
    DATASET_PATH2='./y_train3.npy'
    # VALID_DATA_PATH = './server.npy'
    VALID_DATA_PATH1 = './X_test3.npy'
    VALID_DATA_PATH2 = './y_test3.npy'

    """Start server and train five rounds."""

    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--server_address",
        type=str,
        default=DEFAULT_SERVER_ADDRESS,
        help=f"gRPC server address (default: {DEFAULT_SERVER_ADDRESS})",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=5,
        help="Number of rounds of federated learning (default: 5)",
    )
    parser.add_argument(
        "--sample_fraction",
        type=float,
        default=1,
        help="Fraction of available clients used for fit/evaluate (default: 0.1)",
    )
    parser.add_argument(
        "--min_sample_size",
        type=int,
        default=3,
        help="Minimum number of clients used for fit/evaluate (default: 1)",
    )
    parser.add_argument(
        "--min_num_clients",
        type=int,
        default=3,
        help="Minimum number of available clients required for sampling (default: 2)",
    )
    parser.add_argument(
        "--log_host",
        type=str,
        help="Logserver address (no default)",
    )
    args = parser.parse_args()

    # Load evaluation data
    X_train, X_test, y_train, y_test = modelFunctionsServer.load_data(
        DATASET_PATH1=DATASET_PATH1,DATASET_PATH2=DATASET_PATH2, VALID_DATA_PATH1=VALID_DATA_PATH1,VALID_DATA_PATH2=VALID_DATA_PATH2
    )

    # Create strategy

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=args.sample_fraction,
        min_fit_clients=args.min_sample_size,
        min_available_clients=args.min_num_clients,
        eval_fn=get_eval_fn(X_test,y_test),
        on_fit_config_fn=fit_config,
    )

    # strategy = SaveModelStrategy(
    #     # (same arguments as FedAvg here)
    #     fraction_fit=args.sample_fraction,
    #     min_fit_clients=args.min_sample_size,
    #     min_available_clients=args.min_num_clients,
    #     eval_fn=get_eval_fn(X_test,y_test),
    #     on_fit_config_fn=fit_config,
    # )


    # Configure logger and start server
    fl.common.logger.configure("server", host=args.log_host)
    fl.server.start_server(
        args.server_address,
        config={"num_rounds": args.rounds},
        strategy=strategy,
    )


def fit_config(rnd: int) -> Dict[str, fl.common.Scalar]:
    """Return a configuration with static batch size and (local) epochs."""
    config: Dict[str, fl.common.Scalar] = {
        "epoch_global": str(rnd),
        "epochs": str(1),
        "batch_size": str(64),
    }
    return config


def get_eval_fn(X_test, y_test):
    """Return an evaluation function for centralized evaluation."""

    def evaluate(weights: fl.common.Weights) -> Optional[Tuple[float, float]]:

        model = modelFunctionsServer.load_model((40,30,1),X_test, y_test)
        
        model.set_weights(weights)
        loss, acc = model.evaluate(X_test, y_test, batch_size=len(X_test))
        print(float(acc))
       
       
        return float(loss), float(acc)

    return evaluate


if __name__ == "__main__":
    # class SaveModelStrategy(fl.server.strategy.FedAvg):
    #     def aggregate_fit(
    #         self,
    #         rnd: int,
    #         results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
    #         failures: List[BaseException],
    #     ) -> Optional[fl.common.Weights]:
    #         aggregated_weights = super().aggregate_fit(rnd, results, failures)
    #         if aggregated_weights is not None:
    #             # Save aggregated_weights
    #             print(f"Saving round {rnd} aggregated_weights...")
    #             np.savez(f"round-{rnd}-weights.npz", *aggregated_weights)
    #         return aggregated_weights
    main()
