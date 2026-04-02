"""
This module provides function to measure the execution time for the different PCA methods.
---------------------------------------------------------------------
REQUIRED FUNCTION SIGNATURE
---------------------------------------------------------------------
Any PCA function passed to `measure_time` MUST follow this interface:
    func(X, n_components=..., **kwargs) -> Any
Where:
- X : ndarray of shape (n_samples, n_features)
    Input data matrix
- n_components : int
    Number of principal components to retain
- **kwargs :
    Additional method-specific arguments (optional)
"""

import time
import numpy as np
from typing import Callable, Any, Dict


def measure_time(func: Callable, X: np.ndarray, n_components: int = 2, repeats: int = 5, copy_data: bool = True, return_output: bool = False, **kwargs) -> Any:
    """
    Measure execution time of a PCA implementation.
    Parameters
    ----------
    func : callable
        PCA function:
            func(X, n_components=..., **kwargs)
    X : ndarray
        Input data matrix (n_samples, n_features)
    n_components : int, default=2
        Number of principal components
    repeats : int, default=5
        Number of timed runs
    copy_data : bool, default=True
        If True, copy X before each run to prevent in-place modification
    return_output : bool, default=False
        If True, also return output from last run
    **kwargs : dict
        Additional method-specific arguments

    Returns
    -------
    results : dict
        {"mean_time": float, "median_time": float, "std_time": float, "min_time": float, "max_time": float}
    OR
    (results, output) if return_output=True
    """
    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    times = []
    output = None
    for _ in range(repeats):
        data = X.copy() if copy_data else X
        start = time.perf_counter()
        output = func(data, n_components=n_components, **kwargs)
        end = time.perf_counter()
        times.append(end - start)
    times = np.array(times)
    results = {"mean_time": float(np.mean(times)), "median_time": float(np.median(times)), "std_time": float(np.std(times)), "min_time": float(np.min(times)), "max_time": float(np.max(times))}
    if return_output:
        return results, output
    return results

def print_time_result(method_name: str, results: Dict[str, float]) -> None:
    """
    Print formatted timing results.
    Example
    -------
    >>> print_time_result("SVD PCA", results)
    """
    print(f"\n{method_name}")
    print("-" * 40)
    print(f"Mean Time   : {results['mean_time']:.6f} sec")
    print(f"Median Time : {results['median_time']:.6f} sec")
    print(f"Std Dev     : {results['std_time']:.6f} sec")
    print(f"Min Time    : {results['min_time']:.6f} sec")
    print(f"Max Time    : {results['max_time']:.6f} sec")
    print("-" * 40)


def compare_times(results_dict: Dict[str, Dict[str, float]]) -> None:
    """
    Compare multiple PCA implementations.
    Example
    -------
    >>> compare_times(results)
    """
    print("\nPCA Timing Comparison")
    print("=" * 90)
    print(f"{'Method':<25} {'Mean(s)':<12} {'Median(s)':<12}"
          f"{'Std(s)':<12} {'Min(s)':<12} {'Max(s)':<12}")
    print("=" * 90)

    for name, res in results_dict.items():
        print(f"{name:<25} "
            f"{res['mean_time']:<12.6f} "
            f"{res['median_time']:<12.6f} "
            f"{res['std_time']:<12.6f} "
            f"{res['min_time']:<12.6f} "
            f"{res['max_time']:<12.6f}")
    print("=" * 90)