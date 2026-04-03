"""
Analyses for Randomized PCA and Incremental PCA.

Notes
-----
Performance of Randomized PCA and Incremental PCA is evaluated for different
kinds of data sets with varying no. of samples (n), features (d), and noise.

Analyses performed:
0. correctness (with sklearn implementation as reference),
1. efficiency analysis (time and memory),
2. numerical stability (effect of noise),
3. scaling behaviour (with n and d).

Synthetic data sets used:
- correlated data (correctness, stability, scaling),
- concentric n-dimensional hyperspheres (efficiency analyses).

Author: Suchet Sadekar
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA as SklearnPCA, IncrementalPCA as SklearnIncrementalPCA

from Randomized_PCA import randomized_pca
from incremental_pca import fit as incremental_fit, transform as incremental_transform

from generate_data import get_correlated_data
from generate_hyperspheres import make_concentric_shells
from pca_correctness import evaluate_correctness, print_correctness_result
from pca_memory import measure_memory, print_memory_result
from pca_time import measure_time, print_time_result
from pca_stability import evaluate_stability, print_stability_result
from pca_scaling import evaluate_scaling, plot_scaling


# Wrapper functions (common interface)

def randomized_pca_wrapper(X, n_components=2, **kwargs):
    """Wrapper for manual Randomized PCA to match the common metric-module interface."""
    oversample = kwargs.get("oversample", 5)
    U_k, S_k, Vt_k, evr = randomized_pca(X, k=n_components, oversample=oversample)

    # Projection: X_centered @ Vt_k.T
    X_centered = X - np.mean(X, axis=0)
    X_proj = X_centered @ Vt_k.T

    # Approximate reconstruction
    X_reconstructed = X_proj @ Vt_k + np.mean(X, axis=0)

    return {
        "X_proj": X_proj,
        "explained_variance": S_k ** 2,
        "explained_variance_ratio": evr,
        "components": Vt_k,
        "X_reconstructed": X_reconstructed,
    }


def incremental_pca_wrapper(X, n_components=2, **kwargs):
    """Wrapper for manual Incremental PCA to match the common metric-module interface."""
    batch_size = kwargs.get("batch_size", max(n_components, 256))
    state = incremental_fit(X, n_keep=n_components, batch_size=batch_size)
    X_proj = incremental_transform(X, state)

    # Approximate reconstruction
    X_reconstructed = X_proj @ state["components"] + state["mean"]

    return {
        "X_proj": X_proj,
        "explained_variance": state["explained_variance"],
        "explained_variance_ratio": state["explained_variance_ratio"],
        "components": state["components"],
        "X_reconstructed": X_reconstructed,
    }


# ─────────────────────────────────────────────
# Sklearn reference wrappers
# ─────────────────────────────────────────────

def sklearn_rpca_wrapper(X, n_components=2, **kwargs):
    """Sklearn PCA with svd_solver='randomized' as reference for Randomized PCA."""
    pca = SklearnPCA(n_components=n_components, svd_solver="randomized", random_state=42)
    X_proj = pca.fit_transform(X)
    X_reconstructed = pca.inverse_transform(X_proj)
    return {
        "X_proj": X_proj,
        "explained_variance": pca.explained_variance_,
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "components": pca.components_,
        "X_reconstructed": X_reconstructed,
    }


def sklearn_ipca_wrapper(X, n_components=2, **kwargs):
    """Sklearn IncrementalPCA as reference for Incremental PCA."""
    batch_size = kwargs.get("batch_size", max(n_components, 256))
    ipca = SklearnIncrementalPCA(n_components=n_components, batch_size=batch_size)
    X_proj = ipca.fit_transform(X)
    X_reconstructed = ipca.inverse_transform(X_proj)
    return {
        "X_proj": X_proj,
        "explained_variance": ipca.explained_variance_,
        "explained_variance_ratio": ipca.explained_variance_ratio_,
        "components": ipca.components_,
        "X_reconstructed": X_reconstructed,
    }


# ─────────────────────────────────────────────
# Main execution
# ─────────────────────────────────────────────

if __name__ == "__main__":

    # ── Dataset parameters ──────────────────────────────────────────────────
    n_samples   = 1000
    n_dims      = 50
    noise       = 0.01
    n_components = 2
    n_repeats   = 10   # repetitions for timing / memory

    # Correlated data (used for correctness, stability, scaling)
    corr_data = get_correlated_data(n_samples, n_dims)
    X_corr = corr_data["data"]

    # Concentric hyperspheres (used for efficiency benchmarks)
    X_hyper, _ = make_concentric_shells(
        n_samples=n_samples, n_dims=n_dims, noise=noise
    )

    print("=" * 60)
    print("  RANDOMIZED PCA - ANALYSES")
    print("=" * 60)

    # ── 0. CORRECTNESS ───────────────────────────────────────────────────────
    print("\n[0] Correctness - Randomized PCA vs. Sklearn (randomized SVD)\n")
    results_rpca = evaluate_correctness(
        test_func=randomized_pca_wrapper,
        reference_func=sklearn_rpca_wrapper,
        X=X_corr,
        n_components=n_components,
    )
    print_correctness_result("Manual Randomized PCA vs Sklearn Reference", results_rpca)

    # ── 1. EFFICIENCY ────────────────────────────────────────────────────────
    print("\n[1] Efficiency - Randomized PCA\n")
    time_rpca = measure_time(
        randomized_pca_wrapper, X_hyper,
        n_components=n_components, repeats=n_repeats,
    )
    mem_rpca = measure_memory(
        randomized_pca_wrapper, X_hyper,
        n_components=n_components, repeats=n_repeats,
    )
    label_rpca = (
        f"Randomized PCA runtime\n"
        f"(n-dim hyperspheres; n={n_samples}, d={n_dims}, "
        f"noise={noise}, repeats={n_repeats})"
    )
    print_time_result(label_rpca, time_rpca)
    print_memory_result(
        label_rpca.replace("runtime", "memory usage"), mem_rpca
    )

    # ── 2. NUMERICAL STABILITY ────────────────────────────────────────────────
    print("\n[2] Numerical Stability - Randomized PCA\n")
    noise_range = np.logspace(-6, -1, 6)
    proj_dist_rpca, ev_dist_rpca = [], []

    for epsilon in noise_range:
        res = evaluate_stability(
            randomized_pca_wrapper, X_corr,
            noise_level=epsilon, n_components=n_components,
        )
        print(
            f"  Noise: {epsilon:.2e} | "
            f"Proj dist: {res['projection_distance']:.6f} | "
            f"EVR dist: {res['explained_variance_distance']:.6f}"
        )
        proj_dist_rpca.append(res["projection_distance"])
        ev_dist_rpca.append(res["explained_variance_distance"])

    plt.figure()
    plt.loglog(noise_range, proj_dist_rpca, marker="o", color="SteelBlue",
               label="Projection Distance")
    plt.loglog(noise_range, ev_dist_rpca, marker="s", color="CornflowerBlue",
               label="EVR Distance")
    plt.xlabel("Noise Level (Epsilon)")
    plt.ylabel("Error Metric (Distance)")
    plt.title("Randomized PCA - Numerical Stability (Log-Log)")
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

    # ── 3. SCALING BEHAVIOUR ──────────────────────────────────────────────────
    print("\n[3] Scaling Behaviour - Randomized PCA\n")
    configs_rpca = {
        "Randomized PCA": {"func": randomized_pca_wrapper, "kwargs": {}}
    }

    # Vary n
    sample_sizes = [100, 200, 500, 1000, 2000, 5000, 10_000, 50_000, 100_000]
    sample_results_rpca = evaluate_scaling(
        method_configs=configs_rpca,
        sizes=sample_sizes, vary="samples", n_features=n_dims,
    )
    print("  Visualising scaling with n (sample count):")
    plot_scaling(sample_results_rpca, metric="time",
                 title="Randomized PCA - Runtime vs n", xlabel="No. of samples (n)")
    plot_scaling(sample_results_rpca, metric="memory",
                 title="Randomized PCA - Memory vs n", xlabel="No. of samples (n)")

    # Vary d
    feature_sizes = [2, 5, 10, 50, 100, 500, 1000, 2000]
    feature_results_rpca = evaluate_scaling(
        method_configs=configs_rpca,
        sizes=feature_sizes, vary="features", n_samples=500,
    )
    print("  Visualising scaling with d (dimensionality):")
    plot_scaling(feature_results_rpca, metric="time",
                 title="Randomized PCA - Runtime vs d", xlabel="No. of features (d)")
    plot_scaling(feature_results_rpca, metric="memory",
                 title="Randomized PCA - Memory vs d", xlabel="No. of features (d)")

    # ════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  INCREMENTAL PCA - ANALYSES")
    print("=" * 60)

    # ── 0. CORRECTNESS ───────────────────────────────────────────────────────
    print("\n[0] Correctness - Incremental PCA vs. Sklearn IncrementalPCA\n")
    results_ipca = evaluate_correctness(
        test_func=incremental_pca_wrapper,
        reference_func=sklearn_ipca_wrapper,
        X=X_corr,
        n_components=n_components,
    )
    print_correctness_result(
        "Manual Incremental PCA vs Sklearn Reference", results_ipca
    )

    # ── 1. EFFICIENCY ────────────────────────────────────────────────────────
    print("\n[1] Efficiency - Incremental PCA\n")
    time_ipca = measure_time(
        incremental_pca_wrapper, X_hyper,
        n_components=n_components, repeats=n_repeats,
    )
    mem_ipca = measure_memory(
        incremental_pca_wrapper, X_hyper,
        n_components=n_components, repeats=n_repeats,
    )
    label_ipca = (
        f"Incremental PCA runtime\n"
        f"(n-dim hyperspheres; n={n_samples}, d={n_dims}, "
        f"noise={noise}, repeats={n_repeats})"
    )
    print_time_result(label_ipca, time_ipca)
    print_memory_result(
        label_ipca.replace("runtime", "memory usage"), mem_ipca
    )

    # ── 2. NUMERICAL STABILITY ────────────────────────────────────────────────
    print("\n[2] Numerical Stability - Incremental PCA\n")
    proj_dist_ipca, ev_dist_ipca = [], []

    for epsilon in noise_range:
        res = evaluate_stability(
            incremental_pca_wrapper, X_corr,
            noise_level=epsilon, n_components=n_components,
        )
        print(
            f"  Noise: {epsilon:.2e} | "
            f"Proj dist: {res['projection_distance']:.6f} | "
            f"EVR dist: {res['explained_variance_distance']:.6f}"
        )
        proj_dist_ipca.append(res["projection_distance"])
        ev_dist_ipca.append(res["explained_variance_distance"])

    plt.figure()
    plt.loglog(noise_range, proj_dist_ipca, marker="o", color="DarkSlateBlue",
               label="Projection Distance")
    plt.loglog(noise_range, ev_dist_ipca, marker="s", color="MediumPurple",
               label="EVR Distance")
    plt.xlabel("Noise Level (Epsilon)")
    plt.ylabel("Error Metric (Distance)")
    plt.title("Incremental PCA - Numerical Stability (Log-Log)")
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

    # ── 3. SCALING BEHAVIOUR ──────────────────────────────────────────────────
    print("\n[3] Scaling Behaviour - Incremental PCA\n")
    configs_ipca = {
        "Incremental PCA": {"func": incremental_pca_wrapper, "kwargs": {}}
    }

    # Vary n
    sample_results_ipca = evaluate_scaling(
        method_configs=configs_ipca,
        sizes=sample_sizes, vary="samples", n_features=n_dims,
    )
    print("  Visualising scaling with n (sample count):")
    plot_scaling(sample_results_ipca, metric="time",
                 title="Incremental PCA - Runtime vs n", xlabel="No. of samples (n)")
    plot_scaling(sample_results_ipca, metric="memory",
                 title="Incremental PCA - Memory vs n", xlabel="No. of samples (n)")

    # Vary d
    feature_results_ipca = evaluate_scaling(
        method_configs=configs_ipca,
        sizes=feature_sizes, vary="features", n_samples=500,
    )
    print("  Visualising scaling with d (dimensionality):")
    plot_scaling(feature_results_ipca, metric="time",
                 title="Incremental PCA - Runtime vs d", xlabel="No. of features (d)")
    plot_scaling(feature_results_ipca, metric="memory",
                 title="Incremental PCA - Memory vs d", xlabel="No. of features (d)")
