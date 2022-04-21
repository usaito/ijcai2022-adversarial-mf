import argparse
import warnings

import tensorflow as tf
import yaml

from trainer import Trainer, Tuner

parser = argparse.ArgumentParser()
parser.add_argument("--data", "-d", type=str, required=True)
parser.add_argument("--model_name", "-m", type=str, required=True)
parser.add_argument("--tuning", "-t", action="store_true")

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    tf.get_logger().setLevel("ERROR")
    args = parser.parse_args()
    config = yaml.safe_load(open("../config.yaml", "rb"))

    # load configs
    eta = config["eta"]
    batch_size = config["batch_size"]
    num_steps = config["num_steps"]
    max_iters = config["max_iters"]
    num_sims = config["num_sims"]
    n_trials = config["n_trials"]
    model_name = args.model_name
    tuning = args.tuning
    data = args.data

    # run simulations
    if tuning:
        tuner = Tuner(data=data, model_name=model_name)
        tuner.tune(n_trials=n_trials)

        print("\n", "=" * 25, "\n")
        print(f"Finished Tuning of {model_name}!")
        print("\n", "=" * 25, "\n")

    trainer = Trainer(
        data=data,
        batch_size=batch_size,
        num_steps=num_steps,
        max_iters=max_iters,
        eta=eta,
        model_name=model_name,
    )
    trainer.run_simulations(num_sims=num_sims)

    print("\n", "=" * 25, "\n")
    print(f"Finished Running {model_name}!")
    print("\n", "=" * 25, "\n")
