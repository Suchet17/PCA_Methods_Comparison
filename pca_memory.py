"""
This module provides function to measure the memory usage of different PCA methods.
---------------------------------------------------------------------
REQUIRED FUNCTION SIGNATURE
---------------------------------------------------------------------
Any PCA function passed to `measure_memory` MUST follow this interface:
    func(X, n_components=..., **kwargs) -> Any
Where:
- X : ndarray of shape (n_samples, n_features)
    Input data matrix
- n_components : int
    Number of principal components to retain
- **kwargs :
    Additional method-specific arguments (optional)
This module uses Python's `tracemalloc`, which tracks Python-level memory
allocations. 
"""
import tracemalloc
import numpy as np
from typing import Callable, Any, Dict

def measure_memory(func: Callable, X: np.ndarray, n_components: int = 2, repeats: int = 5, copy_data: bool = True, return_output: bool = False, **kwargs) -> Any:
    """
    Measure memory usage of a PCA implementation.
    Parameters
    ----------
    func : callable
        PCA function following signature:
            func(X, n_components=..., **kwargs)
    X : ndarray
        Input data matrix of shape (n_samples, n_features)
    n_components : int, default=2
        Number of principal components
    repeats : int, default=5
        Number of measured runs
    copy_data : bool, default=True
        Whether to pass X.copy() in each run to avoid in-place modification bias
    return_output : bool, default=False
        If True, also return output from the last run
    **kwargs : dict
        Additional arguments passed to the PCA method

    Returns
    -------
    results : dict
        {"mean_peak_bytes": float, "std_peak_bytes": float, "min_peak_bytes": float, "max_peak_bytes": float, "mean_peak_mib": float, "std_peak_mib": float, "min_peak_mib": float, "max_peak_mib": float}
    OR
        (results, output) if return_output=True
    """
    peak_memories = []
    output = None

    # Measured runs
    for _ in range(repeats):
        data = X.copy() if copy_data else X
        tracemalloc.start()
        output = func(data, n_components=n_components, **kwargs)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_memories.append(peak)
    peak_memories = np.array(peak_memories, dtype=float)
    mib = peak_memories / (1024 ** 2)
    results = {
        "mean_peak_bytes": float(np.mean(peak_memories)),
        "std_peak_bytes": float(np.std(peak_memories)),
        "min_peak_bytes": float(np.min(peak_memories)),
        "max_peak_bytes": float(np.max(peak_memories)),
        "mean_peak_mib": float(np.mean(mib)),
        "std_peak_mib": float(np.std(mib)),
        "min_peak_mib": float(np.min(mib)),
        "max_peak_mib": float(np.max(mib)),
    }

    if return_output:
        return results, output
    return results

def print_memory_result(method_name: str, results: Dict[str, float]) -> None:
    """
    Print memory results for a single PCA method.
    Parameters
    ----------
    method_name : str
        Name of the PCA method
    results : dict
        Output dictionary from `measure_memory`
    """
    print(f"\n{method_name}")
    print("-" * 50)
    print(f"Mean Peak Memory : {results['mean_peak_mib']:.6f} MiB")
    print(f"Std Dev          : {results['std_peak_mib']:.6f} MiB")
    print(f"Min Peak Memory  : {results['min_peak_mib']:.6f} MiB")
    print(f"Max Peak Memory  : {results['max_peak_mib']:.6f} MiB")
    print("-" * 50)

def compare_memory(results_dict: Dict[str, Dict[str, float]]) -> None:
    """
    Compare multiple PCA implementations in tabular format.
    Parameters
    ----------
    results_dict : dict
        Dictionary of method name -> memory results
    """
    print("\nPCA Memory Comparison")
    print("=" * 90)
    print(
        f"{'Method':<25} "
        f"{'Mean(MiB)':<15} "
        f"{'Std(MiB)':<15} "
        f"{'Min(MiB)':<15} "
        f"{'Max(MiB)':<15}")
    print("=" * 90)

    for name, res in results_dict.items():
        print(
            f"{name:<25} "
            f"{res['mean_peak_mib']:<15.6f} "
            f"{res['std_peak_mib']:<15.6f} "
            f"{res['min_peak_mib']:<15.6f} "
            f"{res['max_peak_mib']:<15.6f}")
    print("=" * 90)
