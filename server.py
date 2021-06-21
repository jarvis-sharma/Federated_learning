import flwr as fl


def evaluate_config(rnd: int):
    """Return evaluation configuration dict for each round.
    Perform five local evaluation steps on each client (i.e., use five
    batches) during rounds one to three, then increase to ten local
    evaluation steps.
    """
    val_steps = 5 if rnd < 4 else 10
    return {"val_steps": val_steps}

# Create strategy
strategy = fl.server.strategy.FedAvg(
    # ... other FedAvg agruments
    fraction_eval=0.2,
    min_eval_clients=2,
    min_available_clients=3,
    on_evaluate_config_fn=evaluate_config,
)

# Start Flower server for four rounds of federated learning
fl.server.start_server("[::]:8080", strategy=strategy)




## Start Flower server for three rounds of federated learning
#if __name__ == "__main__":
#    fl.server.start_server("0.0.0.0:8080", config={"num_rounds": 3})
