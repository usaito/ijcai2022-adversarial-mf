from typing import Tuple

import numpy as np
from pandas import DataFrame

from models.onebit_mc import one_bit_MC_fully_observed


def estimate_pscore(
    train: np.ndarray, train_mcar: np.ndarray, val: np.ndarray, model_name: str
) -> Tuple:
    """Estimate pscore."""
    if "item" in model_name:
        pscore = np.unique(train[:, 1], return_counts=True)[1]
        pscore = pscore / pscore.max()
        pscore_train = pscore[train[:, 1]]
        pscore_val = pscore[val[:, 1]]

    elif "user" in model_name:
        pscore = np.unique(train[:, 0], return_counts=True)[1]
        pscore = pscore / pscore.max()
        pscore_train = pscore[train[:, 0]]
        pscore_val = pscore[val[:, 0]]

    elif "both" in model_name:
        user_pscore = np.unique(train[:, 0], return_counts=True)[1]
        user_pscore = user_pscore / user_pscore.max()
        item_pscore = np.unique(train[:, 1], return_counts=True)[1]
        item_pscore = item_pscore / item_pscore.max()
        pscore_train = user_pscore[train[:, 0]] * item_pscore[train[:, 1]]
        pscore_val = user_pscore[val[:, 0]] * item_pscore[val[:, 1]]

    elif "oracle" in model_name:
        pscore = (
            np.unique(train[:, 2], return_counts=True)[1]
            / np.unique(train_mcar[:, 2], return_counts=True)[1]
        )
        pscore = pscore / pscore.max()
        pscore_train = pscore[train[:, 2] - 1]
        pscore_val = pscore[val[:, 2] - 1]

    elif "1bit-mc" in model_name:
        mat = create_obs_matrix(train)
        p_hat = one_bit_MC_fully_observed(mat)
        pscore_train = p_hat[train[:, 0], train[:, 1]]
        pscore_val = p_hat[val[:, 0], val[:, 1]]

    else:  # uniform propensity
        pscore_train = np.ones(train.shape[0])
        pscore_val = np.ones(val.shape[0])

    pscore_train = np.expand_dims(pscore_train, 1)
    pscore_val = np.expand_dims(pscore_val, 1)
    return pscore_train, pscore_val


def create_unlabeled_mcar_data(train: np.ndarray) -> np.ndarray:
    """Create unlabeled MCAR data to be used in the training of MF-DR and DAMF."""
    num_users = train[:, 0].max() + 1
    num_items = train[:, 1].max() + 1
    unlabeled_mcar_data = DataFrame(np.zeros((num_users, num_items)))
    unlabeled_mcar_data = unlabeled_mcar_data.stack().reset_index().values[:, :2]
    # define unlabeled by R_{u,i}=0
    labels = np.zeros(unlabeled_mcar_data.shape[0])
    unlabeled_mcar_data = np.c_[unlabeled_mcar_data, labels]

    return unlabeled_mcar_data


def create_obs_matrix(train: np.ndarray) -> np.ndarray:
    num_users = train[:, 0].max() + 1
    num_items = train[:, 1].max() + 1
    mat = np.zeros((num_users, num_items))
    for u, i in train[:, :2]:
        mat[u, i] = 1
    return mat


def sample_mini_batch_for_damf(
    train: np.ndarray, unlabeled_mcar_data: np.ndarray, batch_size: int
) -> Tuple:
    """Sample mini batch data for DAMF containing
    labeled MNAR and unlabeled MCAR data of equal sizes (=batch_size/2)."""
    batch_size_for_each = np.int(batch_size / 2)
    num_train, num_unlabeled_mcar = train.shape[0], unlabeled_mcar_data.shape[0]
    idx_train = np.random.choice(np.arange(num_train), size=batch_size_for_each)
    idx_unlabeled_mcar = np.random.choice(
        np.arange(num_unlabeled_mcar), size=batch_size_for_each
    )
    labeled_mnar_batch = train[idx_train]
    unlabeled_mcar_batch = unlabeled_mcar_data[idx_unlabeled_mcar]
    labels_batch = np.expand_dims(labeled_mnar_batch[:, 2], 1)

    return labeled_mnar_batch, unlabeled_mcar_batch, labels_batch


def sample_mini_batch_for_mfdr(
    train: np.ndarray,
    pscore: np.ndarray,
    unlabeled_mcar_data: np.ndarray,
    batch_size: int,
) -> Tuple:
    """Sample mini batch data for MF-DR containing
    labeled MNAR and unlabeled MCAR data of equal sizes (=batch_size/2)."""
    batch_size_for_train, batch_size_for_unlabeled = (
        np.int(0.7 * batch_size),
        np.int(0.3 * batch_size),
    )
    num_train, num_unlabeled_mcar = train.shape[0], unlabeled_mcar_data.shape[0]
    idx_train = np.random.choice(np.arange(num_train), size=batch_size_for_train)
    idx_unlabeled_mcar = np.random.choice(
        np.arange(num_unlabeled_mcar), size=batch_size_for_unlabeled
    )
    raw_train_batch = train[idx_train]
    unlabeled_mcar_batch = unlabeled_mcar_data[idx_unlabeled_mcar]
    train_batch = np.r_[raw_train_batch, unlabeled_mcar_batch]
    # define unlabeled by R_{u,i}=0
    labels_batch = np.r_[raw_train_batch[:, 2], np.zeros(batch_size_for_unlabeled)]
    labels_batch = np.expand_dims(labels_batch, 1)
    # define pscores of unlabeled as P_{u,i}=1
    raw_pscore_batch = pscore.flatten()[idx_train]
    pscore_batch = np.r_[raw_pscore_batch, np.ones(batch_size_for_unlabeled)]
    pscore_batch = np.expand_dims(pscore_batch, 1)
    # observation indicators
    obs_batch = np.r_[np.ones(batch_size_for_train), np.zeros(batch_size_for_unlabeled)]
    obs_batch = np.expand_dims(obs_batch, 1)

    return train_batch, labels_batch, pscore_batch, obs_batch


def create_obs_unobs_data(train: np.ndarray, val: np.ndarray) -> Tuple:
    """Create observation-unobservation matrix for
    estimating propensity score by logisticMF"""
    obs_data = np.r_[train, val][:, :2]
    num_users = obs_data[:, 0].max() + 1
    num_items = obs_data[:, 1].max() + 1
    all_data = DataFrame(np.zeros((num_users, num_items)))
    all_data = all_data.stack().reset_index().values[:, :2]
    unobs_data = list(set(map(tuple, all_data)) - set(map(tuple, obs_data)))
    unobs_data = np.array(unobs_data, dtype=int)
    obs_unobs_data = np.r_[
        np.c_[obs_data, np.ones(obs_data.shape[0])],
        np.c_[unobs_data, np.zeros(unobs_data.shape[0])],
    ]

    return num_users, num_items, obs_unobs_data
