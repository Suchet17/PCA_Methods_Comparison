"""
PCA using Singular Value Decomposition (SVD).
PCA by SVD has been implemented and compared to Scikit-learn SVD method.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from generate_data import get_correlated_data
from pca_time import measure_time, compare_times
from pca_memory import measure_memory, compare_memory
from pca_correctness import evaluate_correctness, print_correctness_result
from pca_stability import evaluate_stability, print_stability_result
from pca_scaling import evaluate_scaling

def pca_svd(X, n_components=2):
    """
    Perform PCA using Singular Value Decomposition (SVD).

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Input data matrix.
    n_components : int, default=2
        Number of principal components to retain.

    Returns
    -------
    dict
        Keys:
        - "X_proj" : projected data of shape (n_samples, n_components)
        - "components" : principal axes of shape (n_components, n_features)
        - "explained_variance" : variance captured by each selected component
        - "explained_variance_ratio" : fraction of total variance explained
        - "mean" : feature-wise mean of original data
        - "X_reconstructed" : reconstruction from selected components
        - "model" : lightweight model object/dict for consistency
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")
    n_samples, n_features = X.shape
    if not (1 <= n_components <= min(n_samples, n_features)):
        raise ValueError("n_components must satisfy 1 <= n_components <= min(n_samples, n_features).")
    mean = np.mean(X, axis=0)
    X_centered = X - mean

    # SVD of centered data
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    U_k = U[:, :n_components]
    S_k = S[:n_components]
    Vt_k = Vt[:n_components, :]
    X_proj = U_k * S_k
    components = Vt_k
    explained_variance = (S_k ** 2) / (n_samples - 1)
    full_explained_variance = (S ** 2) / (n_samples - 1)
    total_variance = np.sum(full_explained_variance)
    if total_variance > 0:
        explained_variance_ratio = explained_variance / total_variance
    else:
        explained_variance_ratio = np.zeros_like(explained_variance)

    X_reconstructed = X_proj @ components + mean
    return {"X_proj": X_proj, "components": components, "explained_variance": explained_variance, "explained_variance_ratio": explained_variance_ratio, "mean": mean, "X_reconstructed": X_reconstructed,}

def sklearn_pca_wrapper(X, n_components=2):
    """
    Wrapper around sklearn PCA to keep output format consistent.
    """
    pca = PCA(n_components=n_components)
    X_proj = pca.fit_transform(X)
    return {"X_proj": X_proj, "components": pca.components_, "explained_variance": pca.explained_variance_, "explained_variance_ratio": pca.explained_variance_ratio_, "mean": pca.mean_, "X_reconstructed": pca.inverse_transform(X_proj)}

def align_projection_signs(A, B):
    """
    Align the signs of projected coordinates in A to match B column-wise.
    """
    A = A.copy()
    n_cols = min(A.shape[1], B.shape[1])
    for j in range(n_cols):
        if np.dot(A[:, j], B[:, j]) < 0:
            A[:, j] *= -1
    return A

def add_pc_info_box(ax, result):
    ev = result["explained_variance"]
    evr = result["explained_variance_ratio"]
    lines = []
    for i in range(min(2, len(ev))):
        lines.append(f"PC{i+1} variance = {ev[i]:.4f}")
        lines.append(f"PC{i+1} ratio = {evr[i] * 100:.2f}%")
    text = "\n".join(lines)
    ax.text(0.02, 0.98, text, transform=ax.transAxes, va="top", ha="left", fontsize=9, bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))

def plot_comparison(n, d):
    """
    Compare manual SVD PCA against sklearn PCA visually in 2D.
    Parameters
    ----------
    n : int
        Number of samples.
    d : int
        Number of features.
    """
    data = get_correlated_data(n=n, k=d, noise=0.05, random_state=42)
    X = StandardScaler().fit_transform(data["data"])
    manual = pca_svd(X, n_components=2)
    sk = sklearn_pca_wrapper(X, n_components=2)
    X_manual = align_projection_signs(manual["X_proj"], sk["X_proj"])
    X_sk = sk["X_proj"]
    evr_manual = manual["explained_variance_ratio"]
    evr_sk = sk["explained_variance_ratio"]
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].scatter(X_manual[:, 0], X_manual[:, 1], alpha=0.6)
    ax[0].set_title("Manual SVD PCA")
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

    plt.suptitle(f"SVD PCA Comparison (n={n}, d={d})", fontsize=14)
    plt.tight_layout()
    plt.show()

def run_analysis(n=1000, d=50, n_components=5):
    """
    Run timing, memory, correctness, and stability analysis.
    """
    data = get_correlated_data(n=n, k=d, noise=0.05, random_state=42)
    X = StandardScaler().fit_transform(data["data"])
    print("\n" + "=" * 80)
    print("SVD PCA ANALYSIS")
    print("=" * 80)

    # Time
    time_svd = measure_time(pca_svd, X, n_components=n_components)
    time_sklearn = measure_time(sklearn_pca_wrapper, X, n_components=n_components)
    compare_times({"SVD PCA": time_svd, "Sklearn PCA": time_sklearn})

    # Memory
    mem_svd = measure_memory(pca_svd, X, n_components=n_components)
    mem_sklearn = measure_memory(sklearn_pca_wrapper, X, n_components=n_components)
    compare_memory({"SVD PCA": mem_svd, "Sklearn PCA": mem_sklearn})

    # Correctness
    correctness = evaluate_correctness(pca_svd, sklearn_pca_wrapper, X, n_components=n_components)
    print("\nCorrectness")
    print("=" * 80)
    for k, v in correctness.items():
        print(f"{k:35}: {v}")

    # Stability
    stability = evaluate_stability(pca_svd, X, n_components=n_components, noise_level=0.01)
    print("\nStability")
    print("=" * 80)
    for k, v in stability.items():
        print(f"{k:35}: {v}")
    return {"time": time_svd, "memory": mem_svd, "correctness": correctness, "stability": stability}

def run_scaling(n_components=5, noise=0.05):
    
    """
    Run scaling analysis for manual SVD PCA and sklearn PCA.
    """
    method_configs = {"Manual SVD PCA": {"func": pca_svd, "kwargs": {}}, "Sklearn PCA": {"func": sklearn_pca_wrapper, "kwargs": {}},}

    scaling_samples = evaluate_scaling(method_configs=method_configs, sizes=[100, 300, 500, 1000], vary="samples", n_features=50, n_components=n_components, noise=noise,)
    scaling_features = evaluate_scaling(method_configs=method_configs, sizes=[10, 20, 50, 100], vary="features", n_samples=1000, n_components=n_components, noise=noise,)

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