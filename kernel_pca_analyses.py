"""
Analyses for Kernel PCA.

Notes
-----
Performance of KPCA is evaluated for different kinds of data sets 
 with varying no. of samples (n), features (d), and noise. 

Analyses performed:
 0. correctness (with sklearn implementation as reference),
 1. efficiency analysis (time and memory),
 2. numerical stability (effect of noise),
 3. scaling behaviour (with n and d).

Syntehtic data sets used:
 - concentric n-dimensional hyperspheres (efficiency analyses),
 - correlated data (stability and scaling).

Author
------
Anshita Singh
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import KernelPCA as SklearnKPCA

from kernel_pca import k_pca

from generate_data import get_correlated_data
from generate_hyperspheres import make_concentric_shells
from pca_correctness import evaluate_correctness, print_correctness_result
from pca_memory import measure_memory, print_memory_result
from pca_time import measure_time, print_time_result
from pca_stability import evaluate_stability, print_stability_result
from pca_scaling import evaluate_scaling, plot_scaling

# parameters 
def kpca_wrapper(X, n_components=2, **kwargs):
  """wrapper function to provide common input to metric modules"""
  gamma = kwargs.get('gamma', 2)
  lambdas, alphas, evr = k_pca(X, gamma=gamma, k=n_components)

  return {
        "X_proj": alphas,
        "explained_variance": lambdas,
        "explained_variance_ratio": evr,
        "components": None,       # not applicable for KPCA
        "X_reconstructed": None   # not implemented
    }
    
# execution
if __name__ == "__main__":
 
  # data set (concentric n-dimensional hyperspheres)
  n_samples = 1000
  n_dims = 10
  noise = 0.01
  X, y = make_concentric_shells(n_samples=n_samples, n_dims=n_dims, noise=noise)
  
  # parameters for k_pca
  n_components = 2
  gamma = 2 # has to be adjusted based on dataset 
 
  # 0. CORRECTNESS
  # wrapper for sklearn (reference) implementation
  def sklearn_kpca_wrapper(X, n_components=2, **kwargs):
    gamma = kwargs.get('gamma', 2) 
    transformer = SklearnKPCA(n_components=n_components, kernel='rbf', gamma=gamma)
    X_proj = transformer.fit_transform(X)
    eigenvals = transformer.eigenvalues_

    transformer_all = SklearnKPCA(kernel='rbf', gamma=gamma)
    X_proj_all = transformer_all.fit_transform(X)
    sklearn_evr = eigenvals / np.sum(transformer_all.eigenvalues_)

    return {
        "X_proj": X_proj,
        "explained_variance_ratio": sklearn_evr, 
        "components": None,
        "X_reconstructed": None
    }
  # comparing with sklearn implementation
  results = evaluate_correctness(
    test_func=kpca_wrapper,
    reference_func=sklearn_kpca_wrapper,
    X=X,
    n_components=n_components,
    test_kwargs={'gamma': gamma},
    reference_kwargs={'gamma': gamma})
  print_correctness_result("Manual Kernel PCA vs Sklearn Reference", results)

 
  # 1. EFFICIANCY ANALYSIS
 
  n_repeats = 10 # no. of iterations, default=5 
  time_result = measure_time(kpca_wrapper, X, n_components=n_components, gamma=gamma, repeats=n_repeats)
  mem_result = measure_memory(kpca_wrapper, X, n_components=n_components, gamma=gamma, repeats=n_repeats)
    
  print_time_result(f"Kernel PCA runtime (Dataset: n-dimensional hyperspheres;\nn : {n_samples}, d : {n_dims}, noise : {noise}, repeats : {n_repeats})", time_result)
  print_memory_result(f"Kernel PCA memory usage (Dataset: n-dimensional hyperspheres;\nn : {n_samples}, d : {n_dims}, noise : {noise}, repeats : {n_repeats})", mem_result)


  # 2. NUMERICAL STABILITY
  print("\nEvaluating numerical stability...\n")
  noise_range = np.logspace(-6, -1, 6)
  proj_dist = []
  ev_dist = []

  for epsilon in noise_range:
    stability_result = evaluate_stability(kpca_wrapper, X, noise_level=epsilon, n_components=n_components, gamma=gamma)
    print(f"Noise : {epsilon}\nProjection ditance : {stability_result["projection_distance"]:.6f}\nExplained variance distance : {stability_result["explained_variance_distance"]:.6f}\n")
    # note : other metrics have not been computed for kernel pca

    proj_dist.append(stability_result["projection_distance"])
    ev_dist.append(stability_result["explained_variance_distance"])

  plt.loglog(noise_range, proj_dist, marker='o', c='DarkSlateBlue', label="Projection Distance")
  plt.loglog(noise_range, ev_dist, marker='s', c='DodgerBlue', label="EVR Distance")
  plt.xlabel("Noise Level (Epsilon)")
  plt.ylabel("Error Metric (Distance)")
  plt.title("Numerical Stability Analysis (Log-Log)")
  plt.grid()
  plt.legend()
  plt.show()

  # 3. SCALING BEHAVIOUR
  print("\nEvaluating scaling behaviour...")
  configs = {"Kernel PCA" : {"func": kpca_wrapper, "kwargs": {"gamma" : gamma}}}

  # with n
  sample_sizes = [100, 200, 500, 1000, 2000, 5000]
  n_dims = 50

  sample_results = evaluate_scaling(method_configs=configs, sizes=sample_sizes, vary="samples", n_features=n_dims)
  print("\nVisualisng scaling behaviour as dataset size (n) increases:")
  plot_scaling(sample_results, metric="time", title="Runtime (with varying n)", xlabel="No. of samples (n)")
  plot_scaling(sample_results, metric="memory", title="Memory usage (with varying n)", xlabel="No. of samples (n)")

  # with d
  feature_sizes = [2, 5, 10, 50, 100, 200]
  n_samples = 500
  feature_results = evaluate_scaling(method_configs=configs, sizes=feature_sizes, vary="features", n_samples=n_samples)
  print("\nVisualising scaling behaviour as dimensionality (d) increases:")
  plot_scaling(feature_results, metric="time", title="Runtime (with varying d)", xlabel="No. of features (d)")
  plot_scaling(feature_results, metric="memory", title="Memory usage (with varying d)", xlabel="No. of features (d)") 
