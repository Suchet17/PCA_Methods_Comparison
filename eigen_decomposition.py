"""
PCA by Eigen Decomposition of covariance matrix.
The method has been implemented using numpy and
compared with PCA method already implemented in 
Scikit-learn.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from generate_data import get_correlated_data
from pca_time import measure_time, compare_times
from pca_memory import measure_memory, compare_memory
from pca_correctness import evaluate_correctness
from pca_stability import evaluate_stability
from pca_scaling import evaluate_scaling


def pca_eigendecomposition(X, n_components=2):
    """
    PCA using eigen-decomposition of the covariance matrix.
    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Input data matrix.
    n_components : int, default=2
        Number of principal components to retain.

    Returns
    -------
    dict
        - "X_proj" : projected data of shape (n_samples, n_components)
        - "components" : principal axes of shape (n_components, n_features)
        - "explained_variance" : variance captured by each selected component
        - "explained_variance_ratio" : fraction of total variance explained
        - "mean" : feature-wise mean of original data
        - "X_reconstructed" : reconstruction from selected components
    """
    X = np.asarray(X, dtype=float)

    n_samples = X.shape[0]
    mean = np.mean(X, axis=0)
    X_centered = X - mean
    cov = (X_centered.T @ X_centered) / (n_samples - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    eigenvalues = np.maximum(eigenvalues, 0.0)
    eigenvalues_k = eigenvalues[:n_components]
    eigenvectors_k = eigenvectors[:, :n_components]
    X_proj = X_centered @ eigenvectors_k
    total_var = np.sum(eigenvalues)
    explained_variance_ratio = (eigenvalues_k / total_var if total_var > 0 else np.zeros_like(eigenvalues_k))
    X_reconstructed = X_proj @ eigenvectors_k.T + mean

    return {"X_proj": X_proj, "components": eigenvectors_k.T, "explained_variance": eigenvalues_k, "explained_variance_ratio": explained_variance_ratio, "mean": mean, "X_reconstructed": X_reconstructed,}

def sklearn_pca_wrapper(X, n_components=2):
    """
    Wrapper around sklearn PCA to keep output format consistent.
    """
    pca = PCA(n_components=n_components)
    X_proj = pca.fit_transform(X)
    return {"X_proj": X_proj, "components": pca.components_, "explained_variance": pca.explained_variance_, "explained_variance_ratio": pca.explained_variance_ratio_, "mean": pca.mean_, "X_reconstructed": pca.inverse_transform(X_proj),}

def align_signs(reference_proj, target_proj):
    """
    Align signs of target projected components to reference.
    """
    aligned = target_proj.copy()
    n_components = min(reference_proj.shape[1], target_proj.shape[1])
    for i in range(n_components):
        if np.dot(reference_proj[:, i], aligned[:, i]) < 0:
            aligned[:, i] *= -1
    return aligned

def add_pc_info_box(ax, result):
    ev = result["explained_variance"]
    evr = result["explained_variance_ratio"]
    lines = []
    for i in range(min(2, len(ev))):
        lines.append(f"PC{i+1} var = {ev[i]:.3f}")
        lines.append(f"PC{i+1} ratio = {evr[i]*100:.2f}%")
    text = "\n".join(lines)
    ax.text(0.02, 0.98, text, transform=ax.transAxes, va="top", ha="left", fontsize=9, bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))

def plot_comparison(n, d):
    """
    Plot comparison between manual Eigen PCA and sklearn PCA.
    Parameters
    ----------
    n : int
        Number of samples.
    d : int
        Number of features.
    """
    data = get_correlated_data(n=n, k=d, noise=0.05, random_state=42)
    X = StandardScaler().fit_transform(data["data"])
    manual = pca_eigendecomposition(X, n_components=2)
    sk = sklearn_pca_wrapper(X, n_components=2)
    X_manual = manual["X_proj"]
    X_sk = align_signs(X_manual, sk["X_proj"])
    evr_manual = manual["explained_variance_ratio"]
    evr_sk = sk["explained_variance_ratio"]
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].scatter(X_manual[:, 0], X_manual[:, 1], alpha=0.6)
    ax[0].set_title("Eigen PCA")
    ax[0].set_xlabel(f"PC1 ({evr_manual[0] * 100:.2f}% variance)")
    ax[0].set_ylabel(f"PC2 ({evr_manual[1] * 100:.2f}% variance)")
    ax[0].grid(True, alpha=0.3)
    add_pc_info_box(ax[0], manual)

    ax[1].scatter(X_sk[:, 0], X_sk[:, 1], alpha=0.6)
    ax[1].set_title("Sklearn PCA")
    ax[1].set_xlabel(f"PC1 ({evr_sk[0] * 100:.2f}% variance)")
    ax[1].set_ylabel(f"PC2 ({evr_sk[1] * 100:.2f}% variance)")
    ax[1].grid(True, alpha=0.3)
    add_pc_info_box(ax[1], sk)
    plt.suptitle(f"PCA Comparison for n={n}, d={d}", fontsize=14)
    plt.tight_layout()
    plt.show()


# Analysis
def run_analysis(n=1000, d=50, n_components=5):
    """
    Run timing, memory, correctness, and stability analysis.
    """
    data = get_correlated_data(n=n, k=d, noise=0.05, random_state=42)
    X = StandardScaler().fit_transform(data["data"])

    print("\n" + "=" * 80)
    print("EIGEN PCA ANALYSIS")
    print("=" * 80)

    # Time
    time_eigen = measure_time(pca_eigendecomposition, X, n_components=n_components)
    time_sklearn = measure_time(sklearn_pca_wrapper, X, n_components=n_components)
    compare_times({"Eigen PCA": time_eigen, "Sklearn PCA": time_sklearn})

    # Memory
    mem_eigen = measure_memory(pca_eigendecomposition, X, n_components=n_components)
    mem_sklearn = measure_memory(sklearn_pca_wrapper, X, n_components=n_components)
    compare_memory({"Eigen PCA": mem_eigen, "Sklearn PCA": mem_sklearn})

    # Correctness
    correctness = evaluate_correctness(pca_eigendecomposition, sklearn_pca_wrapper, X, n_components=n_components)
    print("\nCorrectness")
    print("=" * 80)
    for k, v in correctness.items():
        print(f"{k:35}: {v}")

    # Stability
    stability = evaluate_stability(pca_eigendecomposition, X, n_components=n_components, noise_level=0.01)

    print("\nStability")
    print("=" * 80)
    for k, v in stability.items():
        print(f"{k:35}: {v}")

    return {"time": time_eigen, "memory": mem_eigen, "correctness": correctness, "stability": stability}

# Scaling
def run_scaling(n_components=5, noise=0.05):
    """
    Run PCA scaling analysis over varying sample and feature sizes.
    """
    method_configs = {"Eigen PCA": { "func": pca_eigendecomposition,"kwargs": {}}, "Sklearn PCA": {"func": sklearn_pca_wrapper,"kwargs": {}}}

    scaling_samples = evaluate_scaling(method_configs=method_configs, sizes=[100, 300, 500, 1000], vary="samples", n_features=50, n_components=n_components, noise=noise)
    scaling_features = evaluate_scaling(method_configs=method_configs, sizes=[10, 20, 50, 100], vary="features", n_samples=1000, n_components=n_components, noise=noise)

    print("\nScaling Results: varying samples")
    print(scaling_samples)
    print("\nScaling Results: varying features")
    print(scaling_features)
    return scaling_samples, scaling_features

if __name__ == "__main__":
    experiments = [(1000, 500), (10000, 10), (500, 1000), (10, 1000),]
    for n, d in experiments:
        plot_comparison(n, d)
    run_analysis(n=1000, d=50, n_components=5)
    run_scaling(n_components=5, noise=0.05)
    