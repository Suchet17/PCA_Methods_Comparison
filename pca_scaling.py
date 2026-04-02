"""
This module evaluates how different PCA methods scale with:
- Number of samples (n)
- Number of features (k)

It measures:
- Runtime 
- Memory usage
---------------------------------------------------------------------
REQUIRED FUNCTION SIGNATURE
---------------------------------------------------------------------
Each PCA method must follow:
    func(X, n_components=..., **kwargs)
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any
from generate_data import get_correlated_data
from pca_time import measure_time
from pca_memory import measure_memory

def evaluate_scaling(method_configs: Dict[str, Dict[str, Any]], sizes: list, vary: str = "samples", n_features: int = 50, n_samples: int = 1000, n_components: int = 2, noise: float = 0.05,) -> Dict[str, dict]:
    """
    Run scaling experiment for multiple PCA methods.
    Parameters
    ----------
    method_configs : dict
        {"MethodName": {"func": callable,"kwargs": dict}}
    sizes : list
        List of values for scaling (samples or features)
    vary : str, default="samples"
        "samples" or "features"
    n_features : int
        Used when varying samples
    n_samples : int
        Used when varying features
    n_components : int
        Number of principal components
    noise : float
        Noise level in synthetic data

    Returns
    -------
    results : dict
        {method_name: {"sizes": [...], "time": [...], "memory": [...]}}
    """
    results = {}

    for method_name, config in method_configs.items():
        func = config["func"]
        kwargs = config.get("kwargs", {})
        times = []
        memories = []
        for size in sizes:
            # Generate data
            if vary == "samples":
                dataset = get_correlated_data(n=size, k=n_features, noise=noise)
            elif vary == "features":
                dataset = get_correlated_data(n=n_samples, k=size, noise=noise)
            else:
                raise ValueError("vary must be 'samples' or 'features'")
            X = dataset["data"]

            # Measure time
            time_res = measure_time(func, X, n_components=n_components, **kwargs)

            # Measure memory
            mem_res = measure_memory(func, X, n_components=n_components, **kwargs)
            times.append(time_res["mean_time"])
            memories.append(mem_res["mean_peak_mib"])
        results[method_name] = {"sizes": sizes, "time": times, "memory": memories,}
    return results

def plot_scaling(results: Dict[str, dict], metric: str = "time", title: str = None, xlabel: str = "Size",):
    """
    Plot scaling results.
    Parameters
    ----------
    results : dict
        Output from run_scaling_experiment
    metric : str, default="time"
        "time" or "memory"
    title : str, optional
        Plot title
    xlabel : str, default="Size"
        Label for x-axis
    """

    if metric not in ["time", "memory"]:
        raise ValueError("metric must be 'time' or 'memory'")
    plt.figure()
    for method, res in results.items():
        plt.plot(res["sizes"], res[metric], marker="o", label=method)
    plt.xlabel(xlabel)
    if metric == "time":
        plt.ylabel("Runtime (seconds)")
    else:
        plt.ylabel("Memory (MiB)")
    if title:
        plt.title(title)
    else:
        plt.title(f"PCA Scaling ({metric})")
    plt.legend()
    plt.grid(True)
    plt.show()