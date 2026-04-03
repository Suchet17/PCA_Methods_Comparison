"""
Manual Sparse PCA compared against sklearn SparsePCA.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import SparsePCA
from sklearn.preprocessing import StandardScaler
from generate_data import get_correlated_data
from pca_time import measure_time, compare_times
from pca_memory import measure_memory, compare_memory
from pca_correctness import evaluate_correctness
from pca_stability import evaluate_stability
from pca_scaling import evaluate_scaling, plot_scaling

def soft_threshold(x, alpha):
    """
    Apply elementwise soft-thresholding.
    Parameters
    ----------
    x : ndarray
        Input vector.
    alpha : float
        Sparsity threshold. Larger values produce more zeros.
    Returns
    -------
    ndarray
        Thresholded vector.
    """
    return np.sign(x) * np.maximum(np.abs(x) - alpha, 0.0)


def align_projection_signs(A, B):
    """
    Align the signs of projected coordinates in A to match B column-wise.

    PCA-type methods are sign ambiguous, so this makes visual comparison fairer.
    """
    A = A.copy()
    n_cols = min(A.shape[1], B.shape[1])
    for j in range(n_cols):
        if np.dot(A[:, j], B[:, j]) < 0:
            A[:, j] *= -1
    return A

def add_pc_info_box(ax, result):
    """
    Add a small information box showing variance captured by the first
    one or two components.
    """
    ev = result.get("explained_variance", None)
    evr = result.get("explained_variance_ratio", None)
    if ev is None or evr is None:
        text = "Variance information\nnot available"
    else:
        lines = []
        for i in range(min(2, len(ev))):
            lines.append(f"PC{i+1} variance = {ev[i]:.4f}")
            lines.append(f"PC{i+1} ratio = {evr[i] * 100:.2f}%")
        text = "\n".join(lines)
    ax.text(0.02, 0.98, text, transform=ax.transAxes, va="top", ha="left", fontsize=9, bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),)

def sparse_power_component(X, alpha=1.0, max_iter=300, tol=1e-6, random_state=42):
    """
    Extract one sparse principal component using a thresholded power iteration.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Centered data matrix.
    alpha : float, default=1.0
        Sparsity strength. Larger alpha gives sparser components.
    max_iter : int, default=300
        Maximum number of iterations.
    tol : float, default=1e-6
        Convergence tolerance.
    random_state : int, default=42
        Random seed.

    Returns
    -------
    v : ndarray of shape (n_features,)
        Sparse loading vector.
    scores : ndarray of shape (n_samples,)
        Projection of X on the sparse component.
    """
    rng = np.random.default_rng(random_state)
    n_features = X.shape[1]

    v = rng.normal(size=n_features)
    v /= (np.linalg.norm(v) + 1e-12)

    for _ in range(max_iter):
        v_old = v.copy()
        # Power-style update toward a dominant variance direction
        v = X.T @ (X @ v)
        # Impose sparsity
        v = soft_threshold(v, alpha)
        norm_v = np.linalg.norm(v)
        if norm_v < 1e-12:
            v = v_old
            break
        v /= norm_v
        if np.linalg.norm(v - v_old) < tol:
            break
    scores = X @ v
    return v, scores

def pca_sparse_manual(X, n_components=2, alpha=1.0, max_iter=300, tol=1e-6, random_state=42,):
    """
    Perform Sparse PCA using a manual thresholded power-iteration approach
    with sequential deflation.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Input data matrix.
    n_components : int, default=2
        Number of sparse components to extract.
    alpha : float, default=1.0
        Sparsity strength.
    max_iter : int, default=300
        Maximum iterations per component.
    tol : float, default=1e-6
        Convergence tolerance.
    random_state : int, default=42
        Random seed.

    Returns
    -------
    dict
        Keys:
        - "X_proj"
        - "components"
        - "explained_variance"
        - "explained_variance_ratio"
        - "mean"
        - "X_reconstructed"
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")
    n_samples, n_features = X.shape
    if not (1 <= n_components <= min(n_samples, n_features)):
        raise ValueError(
            "n_components must satisfy 1 <= n_components <= min(n_samples, n_features)."
        )
    mean = np.mean(X, axis=0)
    X_centered = X - mean
    X_work = X_centered.copy()
    components = []
    projections = []
    rng = np.random.default_rng(random_state)
    for _ in range(n_components):
        v, scores = sparse_power_component(X_work, alpha=alpha, max_iter=max_iter, tol=tol, random_state=int(rng.integers(0, 1_000_000)),)
        components.append(v)
        projections.append(scores)
        # Deflation
        X_work = X_work - np.outer(scores, v)
    components = np.array(components)                      # (n_components, n_features)
    X_proj = np.column_stack(projections)                 # (n_samples, n_components)
    X_reconstructed = X_proj @ components + mean
    explained_variance = np.var(X_proj, axis=0, ddof=1)
    total_variance = np.var(X_centered, axis=0, ddof=1).sum()

    if total_variance > 0:
        explained_variance_ratio = explained_variance / total_variance
    else:
        explained_variance_ratio = np.zeros_like(explained_variance)

    return {"X_proj": X_proj, "components": components, "explained_variance": explained_variance, "explained_variance_ratio": explained_variance_ratio, "mean": mean, "X_reconstructed": X_reconstructed,}

def sklearn_sparse_pca_wrapper(X, n_components=2, alpha=1.0, random_state=42):
    """
    Wrap sklearn SparsePCA to match the common output dictionary format.
    """
    X = np.asarray(X, dtype=float)
    mean = np.mean(X, axis=0)
    X_centered = X - mean
    model = SparsePCA(
        n_components=n_components,
        alpha=alpha,
        random_state=random_state,
    )

    X_proj = model.fit_transform(X_centered)
    components = model.components_
    X_reconstructed = X_proj @ components + mean
    explained_variance = np.var(X_proj, axis=0, ddof=1)
    total_variance = np.var(X_centered, axis=0, ddof=1).sum()
    if total_variance > 0:
        explained_variance_ratio = explained_variance / total_variance
    else:
        explained_variance_ratio = np.zeros_like(explained_variance)

    return {"X_proj": X_proj, "components": components, "explained_variance": explained_variance, "explained_variance_ratio": explained_variance_ratio, "mean": mean, "X_reconstructed": X_reconstructed,}

def plot_comparison(n, d, noise=0.00, alpha=1.0, random_state=42):
    """
    Compare manual Sparse PCA and sklearn SparsePCA visually in 2D.
    """
    data = get_correlated_data(n=n, k=d, noise=noise, random_state=random_state)
    X = StandardScaler().fit_transform(data["data"])
    manual = pca_sparse_manual(X, n_components=2, alpha=alpha, random_state=random_state,)
    sk = sklearn_sparse_pca_wrapper(X, n_components=2, alpha=alpha, random_state=random_state,)
    X_manual = align_projection_signs(manual["X_proj"], sk["X_proj"])
    X_sk = sk["X_proj"]
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].scatter(X_manual[:, 0], X_manual[:, 1], alpha=0.6)
    ax[0].set_title(f"Manual Sparse PCA (alpha={alpha})")
    ax[0].set_xlabel("Component 1")
    ax[0].set_ylabel("Component 2")
    ax[0].grid(True, alpha=0.3)
    add_pc_info_box(ax[0], manual)

    ax[1].scatter(X_sk[:, 0], X_sk[:, 1], alpha=0.6)
    ax[1].set_title(f"Sklearn SparsePCA (alpha={alpha})")
    ax[1].set_xlabel("Component 1")
    ax[1].set_ylabel("Component 2")
    ax[1].grid(True, alpha=0.3)
    add_pc_info_box(ax[1], sk)

    plt.suptitle(f"Sparse PCA Comparison (n={n}, d={d}, noise={noise}, alpha={alpha})",fontsize=14,)
    plt.tight_layout()
    plt.show()

def run_analysis(n=1000, d=50, n_components=5, alpha=1.0):
    """
    Run timing, memory, correctness, and stability analysis.
    """
    data = get_correlated_data(n=n, k=d, noise=0.05, random_state=42)
    X = StandardScaler().fit_transform(data["data"])
    print("\n" + "=" * 80)
    print("SPARSE PCA ANALYSIS")
    print("=" * 80)
    # Time
    time_manual = measure_time(pca_sparse_manual,X, n_components=n_components, alpha=alpha, random_state=42,)
    time_sklearn = measure_time(sklearn_sparse_pca_wrapper, X, n_components=n_components, alpha=alpha, random_state=42,)
    compare_times({"Manual Sparse PCA": time_manual, "Sklearn SparsePCA": time_sklearn})

    # Memory
    mem_manual = measure_memory(pca_sparse_manual, X, n_components=n_components, alpha=alpha, random_state=42,)
    mem_sklearn = measure_memory(sklearn_sparse_pca_wrapper, X, n_components=n_components, alpha=alpha, random_state=42,)
    compare_memory({"Manual Sparse PCA": mem_manual, "Sklearn SparsePCA": mem_sklearn})

    # Correctness
    correctness = evaluate_correctness(pca_sparse_manual, sklearn_sparse_pca_wrapper, X, n_components=n_components, test_kwargs={"alpha": alpha, "random_state": 42}, reference_kwargs={"alpha": alpha, "random_state": 42}, match_components=True,)
    print("\nCorrectness")
    print("=" * 80)
    for key, value in correctness.items():
        print(f"{key:35}: {value}")

    # Stability
    stability = evaluate_stability(pca_sparse_manual, X, n_components=n_components, noise_level=0.01, alpha=alpha, random_state=42,)
    print("\nStability")
    print("=" * 80)
    for key, value in stability.items():
        print(f"{key:35}: {value}")
    return {"time": time_manual, "memory": mem_manual, "correctness": correctness,"stability": stability,}

def run_scaling(n_components=5, alpha=1.0):
    """
    Run scaling analysis for manual Sparse PCA and sklearn SparsePCA.
    """
    method_configs = {
        "Manual Sparse PCA": {"func": pca_sparse_manual, "kwargs": {"alpha": alpha, "random_state": 42},},}
    #    "Sklearn SparsePCA": {"func": sklearn_sparse_pca_wrapper, "kwargs": {"alpha": alpha, "random_state": 42},},}
    
    scaling_samples = evaluate_scaling(method_configs=method_configs, sizes=[100, 200, 500, 1000, 2000], vary="samples", n_features=10)
    
    plot_scaling(scaling_samples, metric="time", title="Runtime Scaling with Number of Samples", xlabel="Number of samples (n)")
    plot_scaling(scaling_samples, metric="memory", title="Memory Scaling with Number of Samples", xlabel="Number of samples (n)")
    
    scaling_features = evaluate_scaling(method_configs=method_configs, sizes=[2, 5, 10, 20, 50, 100, 200], vary="features", n_samples=500)
    
    plot_scaling(scaling_features, metric="time", title="Runtime Scaling with Number of Features", xlabel="Number of features (d)")
    plot_scaling(scaling_features, metric="memory", title="Memory Scaling with Number of Features", xlabel="Number of features (d)")
    
    print("\nScaling Results: varying samples")
    print(scaling_samples)
    print("\nScaling Results: varying features")
    print(scaling_features)
    return scaling_samples, scaling_features

if __name__ == "__main__":
    #experiments = [(1000, 500), (10000, 10), (500, 1000), (100, 1000),]
    #for n, d in experiments:
    #    plot_comparison(n, d, noise=0.05, alpha=1.0, random_state=42)
    run_analysis(n=1000, d=50, n_components=5, alpha=1.0)
    #run_scaling(n_components=5, alpha=1.0)