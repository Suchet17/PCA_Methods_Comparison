"""
Implementation of Incremental PCA

Author: Suchet Sadekar
"""

import numpy as np
from generate_data import get_lowrank_data, get_correlated_data

def partial_fit(X: np.ndarray, state: dict) -> dict:
    """
    Update the PCA state with one mini-batch X

    Parameters
    ----------
    state : dict
        Current model state containing 'n_keep',
        'mean', 'var', 'seen' (number of samples trained on)
    X : np.ndarray[float] of shape (n, k)
        Data matrix with 'n' rows and 'k' columns

    Returns
    -------
    state : dict (updated in-place and also returned)
    """

    X = np.asarray(X, dtype=float)
    n, k = X.shape
    n_keep = state["n_keep"]

    assert n_keep <= n, f"n_keep ({n_keep}) must be <= n ({n})"

    batch_mean = X.mean(axis=0)
    batch_var = X.var(axis=0)

    if "mean" not in state: #First batch
        state["mean"] = batch_mean
        state["var"] = batch_var
        state["seen"] = n

        X_centered = X - batch_mean
        svd_update(X_centered, state,  n)

    else: # Subsequent batches
        last_mean = state["mean"]
        last_n = state["seen"]
        total_n = last_n + n

        delta = batch_mean - last_mean
        state["mean"] = last_mean + (delta * n / total_n)
        state["var"] = (last_n*state["var"] + (n*batch_var) + (((delta**2)*last_n*n) / total_n)) / total_n
        state["seen"] = total_n

        X_centered = X - batch_mean

        mean_correction = (np.sqrt(last_n * n / total_n) * (last_mean - batch_mean))

        X_temp = np.vstack([state["singular_values"][:, None] * state["components"], X_centered, mean_correction[None, :],])
        svd_update(X_temp, state, total_n)

    return state


def fit(X: np.ndarray, n_keep: int, batch_size: int = -1) -> dict:
    """
    PCA by processing in mini-batches.

    Parameters
    ----------
    X : ndarray of shape (n, k)
    n_keep : int
        number of principal components to keep
    batch_size : int
        rows per mini-batch (default: max(n_keep, 256))

    Returns
    -------
    state : dict with keys:
        components — (n_keep, k) eigenvectors
        singular_values — (n_keep,)
        mean — (k,)  feature means
        var — (k,)  feature variances
        explained_variance — (n_keep,)
        explained_variance_ratio — (n_keep,)
        seen — int
        n_keep — int
    """

    if batch_size == -1:
        batch_size = max(n_keep, 256)
    X = np.asarray(X, dtype=float)
    state = {"n_keep": n_keep}

    for start in range(0, len(X), batch_size):
        partial_fit(X[start : start + batch_size], state)

    return state


def transform(X: np.ndarray, state: dict) -> np.ndarray:
    """
    Project X onto the principal components

    Parameters
    ----------
    state : dict
        Fitted state
    X : ndarray of shape (n, k)
        Data Matrix to transform

    Returns
    -------
    X_transformed : ndarray of shape (n, n_keep)
    """

    if "components" not in state:
        raise RuntimeError("Model has not been fitted yet.")
    X = np.asarray(X, dtype=float)
    return (X - state["mean"]) @ state["components"].T

# Helper function
def svd_update(X_aug: np.ndarray, state: dict, total_n: int) -> None:
    n_keep = state["n_keep"]
    _, S, Vt = np.linalg.svd(X_aug, full_matrices=False)

    state["components"] = Vt[:n_keep]
    state["singular_values"] = S[:n_keep]

    ev = (S[:n_keep] ** 2) / (total_n - 1)
    state["explained_variance"] = ev
    state["explained_variance_ratio"] = ev / np.sum(state["var"])


if __name__ == "__main__":
    n, k = 20_000, 100
    n_keep = k

    #syn_data = get_lowrank_data(n, k, n_keep - 1)
    syn_data = get_correlated_data(n, k)
    X_true, ev_true, pcs_true = syn_data['data'], syn_data['variance'], syn_data['pcs']

    state = fit(X_true, n_keep=n_keep, batch_size=100)
    X_transformed = transform(X_true, state)

    # Full fit
    print(f"explained_variance_ratio : {state['explained_variance_ratio']}")
    print(f"cumulative : {state['explained_variance_ratio'].cumsum()}")
    print(f"projected shape : {X_transformed.shape}")

    # Incremental fits
    state2 = {"n_keep": n_keep}
    for start in range(0, n, 200):
        partial_fit(X_true[start : start + 200], state2)
    X_transformed2 = transform(X_true, state2)

    print(np.max(state["explained_variance"] - state2["explained_variance"]))
    print(np.max(np.abs(state["components"]) - np.abs(state2["components"])))

    assert np.allclose(state["explained_variance"], state2["explained_variance"], rtol=0.1)
    assert np.allclose(state["components"].T @ state["components"],
                       state2["components"].T @ state2["components"], rtol=0.1)
