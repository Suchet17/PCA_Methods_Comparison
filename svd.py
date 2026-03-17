import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from generate_data import get_correlated_data


def pca_svd(X, n_components=2):
    """
    Perform PCA using Singular Value Decomposition (SVD).

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Input data matrix.
    n_components : int, default=2
        Number of principal components.

    Returns
    -------
    X_proj : ndarray
        Projected data.
    explained_variance_ratio : ndarray
        Explained variance ratio.
    """

    X_centered = X - np.mean(X, axis=0)

    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

    X_proj = U[:, :n_components] * S[:n_components]

    explained_variance = (S ** 2) / (X.shape[0] - 1)
    evr = explained_variance[:n_components] / np.sum(explained_variance)

    return X_proj, evr


def make_plot_svd(n, d, noise=0.05, random_state=42):
    data = get_correlated_data(n=n, k=d, noise=noise, random_state=random_state)
    X = data["data"]

    # standardise
    X_std = StandardScaler().fit_transform(X)

    # original data shown using first two raw standardized features
    X_orig = X_std[:, :2]

    # sklearn PCA
    pca = PCA(n_components=2)
    X_sk = pca.fit_transform(X_std)
    evr_sk = pca.explained_variance_ratio_

    # manual SVD PCA
    X_svd, evr_svd = pca_svd(X_std, 2)

    # fix sign ambiguity
    for i in range(2):
        if np.dot(X_sk[:, i], X_svd[:, i]) < 0:
            X_svd[:, i] *= -1

    # determine regime label
    if n > d and n < 10 * d:
        regime = "n > d"
    elif n >= 10 * d:
        regime = "n >> d"
    elif n < d and d < 10 * n:
        regime = "n < d"
    else:
        regime = "n << d"

    fig, ax = plt.subplots(1, 3, figsize=(18, 5))

    ax[0].scatter(X_orig[:, 0], X_orig[:, 1], c="DarkSlateBlue", alpha=0.5, edgecolor="k")
    ax[0].set_title("Original Data")

    ax[1].scatter(X_svd[:, 0], X_svd[:, 1], c="DarkSlateBlue", alpha=0.5, edgecolor="k")
    ax[1].set_title(f"SVD PCA\nPC1: {evr_svd[0]:.2%}, PC2: {evr_svd[1]:.2%}")

    ax[2].scatter(X_sk[:, 0], X_sk[:, 1], c="DarkSlateBlue", alpha=0.5, edgecolor="k")
    ax[2].set_title(f"Sklearn PCA\nPC1: {evr_sk[0]:.2%}, PC2: {evr_sk[1]:.2%}")

    ax[0].set_xlabel("Feature 1")
    ax[0].set_ylabel("Feature 2")

    ax[1].set_xlabel("PC1")
    ax[1].set_ylabel("PC2")

    ax[2].set_xlabel("PC1")
    ax[2].set_ylabel("PC2")

    fig.suptitle(
        f"SVD PCA Comparison ({regime})\n"
        f"n={n}, d={d}, noise={noise}"
    )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    experiments = [
        (1000, 500),
        (10000, 10),
        (500, 1000),
        (10, 1000),
    ]

    for n, d in experiments:
        make_plot_svd(n, d)
