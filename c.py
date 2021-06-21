import json
import numpy as np
import random
import math
import tensorflow.keras as keras
import tensorflow as tf
import os
import flwr as fl
from typing import List
from typing import Tuple



class SaveModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[BaseException],
    ) -> [fl.common.Weights]:
        aggregated_weights = super().aggregate_fit(rnd, results, failures)
        if aggregated_weights is not None:
            # Save aggregated_weights
            print(f"Saving round {rnd} aggregated_weights...")
            np.savez(f"round-{rnd}-weights.npz", *aggregated_weights)
        
        return aggregated_weights

def evaluate_config(rnd: int):
    """Return evaluation configuration dict for each round.
    Perform five local evaluation steps on each client (i.e., use five
    batches) during rounds one to three, then increase to ten local
    evaluation steps.
    """
    val_steps = 5 if rnd < 4 else 10
    return {"val_steps": val_steps}

# Create strategy and run server
strategy = SaveModelStrategy(
    # (same arguments as FedAvg here)
    fraction_eval=0.2,
    min_eval_clients=2,
    min_available_clients=2,
    on_evaluate_config_fn=evaluate_config,
)
for i in range(2):
	fl.server.start_server(strategy=strategy)

