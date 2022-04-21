import time
from pathlib import Path
from typing import Tuple

import numpy as np
import optuna
import pandas as pd
import tensorflow as tf
import yaml
from optuna.samplers import TPESampler
from optuna.trial import Trial
from pandas import DataFrame
from tensorflow.python.framework import ops

from evaluate.evaluator import aoa_evaluator
from models.models import DAMF, MFDR, MFIPS, CausE
from utils.preprocessor import preprocess_yahoo_coat
from utils.train_helpers import (create_unlabeled_mcar_data, estimate_pscore,
                                 sample_mini_batch_for_damf,
                                 sample_mini_batch_for_mfdr)


def train_mfips(
    sess: tf.Session,
    model: MFIPS,
    data: str,
    train: np.ndarray,
    train_mcar: np.ndarray,
    val: np.ndarray,
    test: np.ndarray,
    max_iters: int = 2500,
    batch_size: int = 1024,
    model_name: str = "oracle",
    seed: int = 0,
) -> Tuple:
    """Train and evaluate the MF-IPS models."""
    train_loss_list = []
    val_loss_list = []
    test_mse_list = []

    # initialise all the TF variables
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    num_train = train.shape[0]
    pscore_train, pscore_val = estimate_pscore(
        train=train, train_mcar=train_mcar, val=val, model_name=model_name
    )
    labels_train = np.expand_dims(train[:, 2], 1)
    labels_val = np.expand_dims(val[:, 2], 1)
    labels_test = np.expand_dims(test[:, 2], 1)

    np.random.seed(12345)
    for _ in np.arange(max_iters):
        # mini-batch sampling
        idx = np.random.choice(np.arange(num_train), size=batch_size)
        train_batch, pscore_batch = train[idx], pscore_train[idx]
        labels_batch = labels_train[idx]
        # update model parameters
        _, loss = sess.run(
            [model.apply_grads, model.loss],
            feed_dict={
                model.users: train_batch[:, 0],
                model.items: train_batch[:, 1],
                model.labels: labels_batch,
                model.scores: pscore_batch,
            },
        )
        train_loss_list.append(loss)
        # calculate validation loss
        val_loss = sess.run(
            model.ips_mse,
            feed_dict={
                model.users: val[:, 0],
                model.items: val[:, 1],
                model.labels: labels_val,
                model.scores: pscore_val,
            },
        )
        val_loss_list.append(val_loss)
        # calculate test loss
        test_mse = sess.run(
            model.mse,
            feed_dict={
                model.users: test[:, 0],
                model.items: test[:, 1],
                model.labels: labels_test,
            },
        )
        test_mse_list.append(test_mse)

    # save train, val and test loss curves.
    path = Path(f"../logs/{data}/{model_name}/loss")
    path.mkdir(parents=True, exist_ok=True)
    np.save(file=str(path / f"train_{seed}.npy"), arr=train_loss_list)
    np.save(file=str(path / f"val_{seed}.npy"), arr=val_loss_list)
    np.save(file=str(path / f"test_{seed}.npy"), arr=test_mse_list)
    # obtain user-item embeddings
    u_emb, i_emb, i_bias = sess.run(
        [model.user_embeddings, model.item_embeddings, model.item_b]
    )

    sess.close()

    return (
        np.min(val_loss_list),
        test_mse_list[np.argmin(val_loss_list)],
        u_emb,
        i_emb,
        i_bias,
    )


def train_mfdr(
    sess: tf.Session,
    model: MFDR,
    data: str,
    train: np.ndarray,
    train_mcar: np.ndarray,
    val: np.ndarray,
    test: np.ndarray,
    max_iters: int = 2500,
    batch_size: int = 1024,
    num_steps: int = 1,
    model_name: str = "oracle",
    seed: int = 0,
) -> Tuple:
    """Train and evaluate the MF-DR models."""
    train_loss_list = []
    val_loss_list = []
    test_mse_list = []

    # initialise all the TF variables
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    num_train = train.shape[0]
    pscore_train, pscore_val = estimate_pscore(
        train=train, train_mcar=train_mcar, val=val, model_name=model_name
    )
    labels_train = np.expand_dims(train[:, 2], 1)
    labels_val = np.expand_dims(val[:, 2], 1)
    labels_test = np.expand_dims(test[:, 2], 1)
    unlabeled_mcar_data = create_unlabeled_mcar_data(train=train)

    np.random.seed(12345)
    max_iters = int(max_iters / num_steps)
    for _ in np.arange(max_iters):
        # update model parameters of the imputation model
        for _ in np.arange(num_steps):
            # mini-batch sampling
            idx = np.random.choice(np.arange(num_train), size=batch_size)
            train_batch, pscore_batch = train[idx], pscore_train[idx]
            labels_batch = labels_train[idx]
            # update model parameters of the imputation model
            _ = sess.run(
                model.apply_grads_imp,
                feed_dict={
                    model.users: train_batch[:, 0],
                    model.items: train_batch[:, 1],
                    model.labels: labels_batch,
                    model.scores: pscore_batch,
                },
            )
        # update model parameters of the final prediction model
        for _ in np.arange(num_steps):
            (
                train_batch,
                labels_batch,
                pscore_batch,
                obs_batch,
            ) = sample_mini_batch_for_mfdr(
                train=train,
                pscore=pscore_train,
                unlabeled_mcar_data=unlabeled_mcar_data,
                batch_size=batch_size,
            )
            # update model parameters
            _, loss = sess.run(
                [model.apply_grads, model.loss],
                feed_dict={
                    model.users: train_batch[:, 0],
                    model.items: train_batch[:, 1],
                    model.labels: labels_batch,
                    model.observes: obs_batch,
                    model.scores: pscore_batch,
                },
            )
            train_loss_list.append(loss)
            # calculate validation loss
            val_loss = sess.run(
                model.ips_mse,
                feed_dict={
                    model.users: val[:, 0],
                    model.items: val[:, 1],
                    model.scores: pscore_val,
                    model.labels: labels_val,
                },
            )
            val_loss_list.append(val_loss)
            # calculate test loss
            test_mse = sess.run(
                model.mse,
                feed_dict={
                    model.users: test[:, 0],
                    model.items: test[:, 1],
                    model.labels: labels_test,
                },
            )
            test_mse_list.append(test_mse)

    # save train, val and test loss curves.
    path = Path(f"../logs/{data}/{model_name}/loss")
    path.mkdir(parents=True, exist_ok=True)
    np.save(file=str(path / f"train_{seed}.npy"), arr=train_loss_list)
    np.save(file=str(path / f"val_{seed}.npy"), arr=val_loss_list)
    np.save(file=str(path / f"test_{seed}.npy"), arr=test_mse_list)
    # obtain user-item embeddings
    u_emb, i_emb, i_bias = sess.run(
        [model.user_embeddings, model.item_embeddings, model.item_b]
    )

    sess.close()

    return (
        np.min(val_loss_list),
        test_mse_list[np.argmin(val_loss_list)],
        u_emb,
        i_emb,
        i_bias,
    )


def train_cause(
    sess: tf.Session,
    model: CausE,
    data: str,
    train_mnar: np.ndarray,
    train_mcar: np.ndarray,
    val: np.ndarray,
    test: np.ndarray,
    max_iters: int = 2500,
    batch_size: int = 1024,
    seed: int = 0,
) -> Tuple:
    """Train and evaluate CausE."""
    train_loss_list = []
    val_loss_list = []
    test_mse_list = []

    # initialise all the TF variables
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    num_train_mnar = train_mnar.shape[0]
    labels_train_mnar = np.expand_dims(train_mnar[:, 2], 1)
    labels_train_mcar = np.expand_dims(train_mcar[:, 2], 1)
    labels_val = np.expand_dims(val[:, 2], 1)
    labels_test = np.expand_dims(test[:, 2], 1)

    np.random.seed(12345)
    for _ in np.arange(max_iters):
        # MCAR model
        _ = sess.run(
            model.apply_grads_mcar,
            feed_dict={
                model.users: train_mcar[:, 0],
                model.items: train_mcar[:, 1],
                model.labels: labels_train_mcar,
            },
        )
        # MNAR model
        # mini-batch sampling
        idx = np.random.choice(np.arange(num_train_mnar), size=batch_size)
        train_batch, labels_batch = train_mnar[idx], labels_train_mnar[idx]
        # update model parameters
        _, loss = sess.run(
            [model.apply_grads_mnar, model.loss_mnar],
            feed_dict={
                model.users: train_batch[:, 0],
                model.items: train_batch[:, 1],
                model.labels: labels_batch,
            },
        )
        train_loss_list.append(loss)
        # calculate validation loss
        val_loss = sess.run(
            model.mse,
            feed_dict={
                model.users: val[:, 0],
                model.items: val[:, 1],
                model.labels: labels_val,
            },
        )
        val_loss_list.append(val_loss)
        # calculate test loss
        test_mse = sess.run(
            model.mse,
            feed_dict={
                model.users: test[:, 0],
                model.items: test[:, 1],
                model.labels: labels_test,
            },
        )
        test_mse_list.append(test_mse)

    # save train, val and test loss curves.
    path = Path(f"../logs/{data}/cause/loss")
    path.mkdir(parents=True, exist_ok=True)
    np.save(file=str(path / f"train_{seed}.npy"), arr=train_loss_list)
    np.save(file=str(path / f"val_{seed}.npy"), arr=val_loss_list)
    np.save(file=str(path / f"test_{seed}.npy"), arr=test_mse_list)
    # obtain user-item embeddings
    u_emb, i_emb, i_bias = sess.run(
        [model.user_embeddings_mnar, model.item_embeddings_mnar, model.item_b]
    )

    sess.close()

    return (
        np.min(val_loss_list),
        test_mse_list[np.argmin(val_loss_list)],
        u_emb,
        i_emb,
        i_bias,
    )


def train_damf(
    sess: tf.Session,
    model: DAMF,
    data: str,
    train: np.ndarray,
    val: np.ndarray,
    test: np.ndarray,
    max_iters: int = 2500,
    batch_size: int = 1024,
    num_steps: int = 1,
    seed: int = 0,
    model_name: str = "damf",
) -> Tuple:
    """Train and evaluate the DAMF model."""
    train_loss_list = []
    val_loss_list = []
    upper_bound_list = []
    test_mse_list = []

    # initialise all the TF variables
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    labels_val = np.expand_dims(val[:, 2], 1)
    labels_test = np.expand_dims(test[:, 2], 1)
    unlabeled_mcar_data = create_unlabeled_mcar_data(train=train)

    np.random.seed(12345)
    max_iters = int(max_iters / num_steps)
    for _ in np.arange(max_iters):
        # update model parameters of final prediction model
        for _ in np.arange(num_steps):
            # mini-batch sampling
            (
                labeled_mnar_batch,
                unlabeled_mcar_batch,
                labels_batch,
            ) = sample_mini_batch_for_damf(
                train=train,
                batch_size=batch_size,
                unlabeled_mcar_data=unlabeled_mcar_data,
            )
            # update model parameters
            _, loss = sess.run(
                [model.apply_grads, model.loss],
                feed_dict={
                    model.users: labeled_mnar_batch[:, 0],
                    model.items: labeled_mnar_batch[:, 1],
                    model.users_mcar: unlabeled_mcar_batch[:, 0],
                    model.items_mcar: unlabeled_mcar_batch[:, 1],
                    model.labels: labels_batch,
                },
            )
            train_loss_list.append(loss)
            # calculate the validation loss
            val_loss, upper_bound = sess.run(
                [model.mse, model.val_loss],
                feed_dict={
                    model.users: val[:, 0],
                    model.items: val[:, 1],
                    model.users_mcar: unlabeled_mcar_batch[:, 0],
                    model.items_mcar: unlabeled_mcar_batch[:, 1],
                    model.labels: labels_val,
                },
            )
            val_loss_list.append(val_loss)
            upper_bound_list.append(upper_bound)
            # calculate the test loss
            test_mse = sess.run(
                model.mse,
                feed_dict={
                    model.users: test[:, 0],
                    model.items: test[:, 1],
                    model.labels: labels_test,
                },
            )
            test_mse_list.append(test_mse)

        # approximate Propensity Matrix Divergence (PMD)
        for _ in np.arange(num_steps):
            # mini-batch sampling
            (
                labeled_mnar_batch,
                unlabeled_mcar_batch,
                labels_batch,
            ) = sample_mini_batch_for_damf(
                train=train,
                batch_size=batch_size,
                unlabeled_mcar_data=unlabeled_mcar_data,
            )
            # update model parameters
            _ = sess.run(
                model.apply_grads_pmd,
                feed_dict={
                    model.users: labeled_mnar_batch[:, 0],
                    model.items: labeled_mnar_batch[:, 1],
                    model.users_mcar: unlabeled_mcar_batch[:, 0],
                    model.items_mcar: unlabeled_mcar_batch[:, 1],
                },
            )

    # save train, val and test loss curves.
    path = Path(f"../logs/{data}/{model_name}/loss")
    path.mkdir(parents=True, exist_ok=True)
    np.save(file=str(path / f"train_{seed}.npy"), arr=train_loss_list)
    np.save(file=str(path / f"val_{seed}.npy"), arr=val_loss_list)
    np.save(file=str(path / f"test_{seed}.npy"), arr=test_mse_list)
    np.save(file=str(path / f"upper_bound_{seed}.npy"), arr=upper_bound_list)
    # obtain user-item embeddings
    u_emb, i_emb, i_bias = sess.run(
        [model.user_embeddings, model.item_embeddings, model.item_b]
    )

    sess.close()

    return (
        np.min(val_loss_list),
        test_mse_list[np.argmin(val_loss_list)],
        u_emb,
        i_emb,
        i_bias,
    )


class Objective:
    def __init__(self, data: str, model_name: str = "mf") -> None:
        """Initialize Class"""
        self.data = data
        self.model_name = model_name

    def __call__(self, trial: Trial) -> float:
        """Calculate an objective value."""
        train, train_mcar, val, test, num_users, num_items = preprocess_yahoo_coat(
            data=self.data
        )

        # sample a set of hyperparameters.
        config = yaml.safe_load(open("../config.yaml", "rb"))
        eta = config["eta"]
        max_iters = config["max_iters"]
        batch_size = config["batch_size"]
        num_steps = config["num_steps"]
        dim = np.int(trial.suggest_discrete_uniform("dim", 5, 40, 5))
        if self.data == "coat":
            lam = trial.suggest_loguniform("lam", 1e-4, 1)
            if self.model_name in ("cause", "damf"):
                domain_pen = trial.suggest_loguniform("domain_pen", 1e-2, 1)
        else:
            lam = trial.suggest_loguniform("lam", 1e-8, 1e-2)
            if self.model_name == "damf":
                domain_pen = trial.suggest_loguniform("domain_pen", 1e-2, 1)
            elif self.model_name == "cause":
                domain_pen = trial.suggest_loguniform("domain_pen", 1e-10, 1)

        ops.reset_default_graph()
        sess = tf.Session()
        if self.model_name == "damf":
            damf = DAMF(
                num_users=num_users,
                num_items=num_items,
                dim=dim,
                lam=lam,
                domain_pen=domain_pen,
                eta=eta,
            )
            val_score = train_damf(
                sess,
                model=damf,
                data=self.data,
                train=train,
                val=val,
                test=test,
                num_steps=num_steps,
                max_iters=max_iters,
                batch_size=batch_size,
                model_name=self.model_name,
            )[0]
        elif "ips" in self.model_name:
            mfips = MFIPS(
                num_users=num_users, num_items=num_items, dim=dim, lam=lam, eta=eta
            )
            val_score = train_mfips(
                sess,
                model=mfips,
                data=self.data,
                train=train,
                train_mcar=train_mcar,
                val=val,
                test=test,
                max_iters=max_iters,
                batch_size=batch_size,
                model_name=self.model_name,
            )[0]
        elif "dr" in self.model_name:
            mfdr = MFDR(
                num_users=num_users, num_items=num_items, dim=dim, lam=lam, eta=eta
            )
            val_score = train_mfdr(
                sess,
                model=mfdr,
                data=self.data,
                train=train,
                train_mcar=train_mcar,
                val=val,
                test=test,
                num_steps=num_steps,
                max_iters=max_iters,
                batch_size=batch_size,
                model_name=self.model_name,
            )[0]
        elif "cause" in self.model_name:
            cause = CausE(
                num_users=num_users,
                num_items=num_items,
                dim=dim,
                lam=lam,
                domain_pen=domain_pen,
                eta=eta,
            )
            val_score = train_cause(
                sess,
                model=cause,
                data=self.data,
                train_mnar=train,
                train_mcar=train_mcar,
                val=val,
                test=test,
                max_iters=max_iters,
                batch_size=batch_size,
            )[0]
        return val_score


class Tuner:
    def __init__(self, data: str, model_name: str) -> None:
        """Initialize Class."""
        self.data = data
        self.model_name = model_name

    def tune(self, n_trials: int = 30) -> None:
        """Hyperparameter Tuning by TPE."""
        path = Path(f"../logs/{self.data}")
        path.mkdir(exist_ok=True, parents=True)
        path_model = Path(f"../logs/{self.data}/{self.model_name}")
        path_model.mkdir(exist_ok=True, parents=True)
        # tune hyperparameters by Optuna
        objective = Objective(data=self.data, model_name=self.model_name)
        study = optuna.create_study(sampler=TPESampler(seed=123))
        study.optimize(objective, n_trials=n_trials)
        # save tuning results
        study.trials_dataframe().to_csv(str(path_model / "tuning_results.csv"))
        if Path("../hyper_params.yaml").exists():
            pass
        else:
            yaml.dump(
                dict(yahoo=dict(), coat=dict()),
                open("../hyper_params.yaml", "w"),
                default_flow_style=False,
            )
        time.sleep(np.random.rand())
        hyper_params_dict = yaml.safe_load(open("../hyper_params.yaml", "r"))
        hyper_params_dict[self.data][self.model_name] = study.best_params
        yaml.dump(
            hyper_params_dict,
            open("../hyper_params.yaml", "w"),
            default_flow_style=False,
        )


class Trainer:
    def __init__(
        self,
        data: str,
        batch_size: int = 1024,
        max_iters: int = 2500,
        eta: float = 1e-3,
        num_steps: int = 1,
        model_name: str = "mf",
    ) -> None:
        """Initialize class."""
        self.data = data
        hyper_params = yaml.safe_load(open(f"../hyper_params.yaml", "r"))[data][
            model_name
        ]
        self.dim = np.int(hyper_params["dim"])
        self.lam = hyper_params["lam"]
        if model_name in ("damf", "cause"):
            self.domain_pen = hyper_params["domain_pen"]
        self.batch_size = batch_size
        self.max_iters = max_iters
        self.eta = eta
        self.num_steps = num_steps
        self.model_name = model_name

    def run_simulations(self, num_sims: int = 15) -> None:
        """Train mf."""
        results_mse = []
        results_ndcg = []
        results_recall = []

        path = Path(f"../logs/{self.data}/{self.model_name}")
        path.mkdir(parents=True, exist_ok=True)
        for seed in np.arange(num_sims):
            (
                train,
                train_mcar,
                val,
                test,
                num_users,
                num_items,
            ) = preprocess_yahoo_coat(data=self.data, seed=seed)

            ops.reset_default_graph()
            sess = tf.Session()
            tf.set_random_seed(seed)
            if self.model_name == "damf":
                damf = DAMF(
                    num_users=num_users,
                    num_items=num_items,
                    dim=self.dim,
                    lam=self.lam,
                    domain_pen=self.domain_pen,
                    eta=self.eta,
                )
                _, mse, u_emb, i_emb, i_bias = train_damf(
                    sess,
                    model=damf,
                    data=self.data,
                    train=train,
                    val=val,
                    test=test,
                    num_steps=self.num_steps,
                    max_iters=self.max_iters,
                    batch_size=self.batch_size,
                    seed=seed,
                    model_name=self.model_name,
                )
            elif "ips" in self.model_name:
                mfips = MFIPS(
                    num_users=num_users,
                    num_items=num_items,
                    dim=self.dim,
                    lam=self.lam,
                    eta=self.eta,
                )
                _, mse, u_emb, i_emb, i_bias = train_mfips(
                    sess,
                    model=mfips,
                    data=self.data,
                    train=train,
                    train_mcar=train_mcar,
                    val=val,
                    test=test,
                    max_iters=self.max_iters,
                    batch_size=self.batch_size,
                    seed=seed,
                    model_name=self.model_name,
                )
            elif "dr" in self.model_name:
                mfdr = MFDR(
                    num_users=num_users,
                    num_items=num_items,
                    dim=self.dim,
                    lam=self.lam,
                    eta=self.eta,
                )
                _, mse, u_emb, i_emb, i_bias = train_mfdr(
                    sess,
                    model=mfdr,
                    data=self.data,
                    train=train,
                    train_mcar=train_mcar,
                    val=val,
                    test=test,
                    num_steps=self.num_steps,
                    max_iters=self.max_iters,
                    batch_size=self.batch_size,
                    seed=seed,
                    model_name=self.model_name,
                )
            elif self.model_name == "cause":
                cause = CausE(
                    num_users=num_users,
                    num_items=num_items,
                    dim=self.dim,
                    lam=self.lam,
                    eta=self.eta,
                    domain_pen=self.domain_pen,
                )
                _, mse, u_emb, i_emb, i_bias = train_cause(
                    sess,
                    model=cause,
                    data=self.data,
                    train_mnar=train,
                    train_mcar=train_mcar,
                    val=val,
                    test=test,
                    max_iters=self.max_iters,
                    batch_size=self.batch_size,
                    seed=seed,
                )
            results_mse.append(mse)
            results = aoa_evaluator(
                user_embed=u_emb,
                item_embed=i_emb,
                item_bias=i_bias,
                test=test,
                model_name=self.model_name,
            )
            results_ndcg.append(results.loc["nDCG", self.model_name])
            results_recall.append(results.loc["Recall", self.model_name])
            print(f"{self.model_name}: #{seed+1}...")

        pd.concat(
            [
                DataFrame(results_mse, columns=["MSE"]),
                DataFrame(results_ndcg, columns=["nDCG"]),
                DataFrame(results_recall, columns=["Recall"]),
            ],
            1,
        ).to_csv(str(path / "results.csv"))
