"""
This module evaluates how stable a PCA method is under small perturbations
(noise) in the input data.

---------------------------------------------------------------------
REQUIRED FUNCTION SIGNATURE
---------------------------------------------------------------------
    func(X, n_components=..., **kwargs) -> dict

Expected keys in returned dict:
    {
        "X_proj": ndarray,
        "components": ndarray,
        "explained_variance": ndarray,
        "explained_variance_ratio": ndarray,
        "mean": float,
        "X_reconstructed": ndarray
    }

---------------------------------------------------------------------
METRICS COMPUTED
---------------------------------------------------------------------
- projection_distance
- component_distance
- subspace_similarity
- explained_variance_distance
- reconstruction_error_delta

"""

import numpy as np
from typing import Callable, Dict, Any

# Helpers 
def _safe_get(result: Dict[str, Any], key: str):
    return result.get(key, None)

def _align_signs(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A = A.copy()
    n = min(A.shape[0], B.shape[0])
    for i in range(n):
        if np.dot(A[i], B[i]) < 0:
            A[i] *= -1
    return A

def _align_projection_signs(Xp1: np.ndarray, Xp2: np.ndarray) -> np.ndarray:
    """
    Align projection signs column-wise so sign flips do not inflate distance.
    """
    Xp1 = Xp1.copy()
    n = min(Xp1.shape[1], Xp2.shape[1])
    for j in range(n):
        if np.dot(Xp1[:, j], Xp2[:, j]) < 0:
            Xp1[:, j] *= -1
    return Xp1

def _component_distance(A: np.ndarray, B: np.ndarray) -> float:
    n = min(A.shape[0], B.shape[0])
    A = _align_signs(A[:n], B[:n])
    B = B[:n]

    denom = np.linalg.norm(B, ord="fro")
    if denom == 0:
        return float(np.linalg.norm(A - B, ord="fro"))
    return float(np.linalg.norm(A - B, ord="fro") / denom)

def _projection_distance(A: np.ndarray, B: np.ndarray) -> float:
    n_cols = min(A.shape[1], B.shape[1])
    A = _align_projection_signs(A[:, :n_cols], B[:, :n_cols])
    B = B[:, :n_cols]

    denom = np.linalg.norm(B, ord="fro")
    if denom == 0:
        return float(np.linalg.norm(A - B, ord="fro"))
    return float(np.linalg.norm(A - B, ord="fro") / denom)

def _component_similarity(A: np.ndarray, B: np.ndarray) -> float:
    n = min(A.shape[0], B.shape[0])
    A = _align_signs(A[:n], B[:n])
    B = B[:n]

    sims = []
    for a, b in zip(A, B):
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na == 0 or nb == 0:
            sims.append(0.0)
        else:
            sims.append(abs(np.dot(a, b) / (na * nb)))
    return float(np.mean(sims)) if sims else None

def _explained_variance_distance(A: np.ndarray, B: np.ndarray) -> float:
    n = min(len(A), len(B))
    A = A[:n]
    B = B[:n]

    denom = np.linalg.norm(B)
    if denom == 0:
        return float(np.linalg.norm(A - B))
    return float(np.linalg.norm(A - B) / denom)

def _reconstruction_error(X: np.ndarray, X_rec: np.ndarray) -> float:
    denom = np.linalg.norm(X, ord="fro")
    if denom == 0:
        return 0.0
    return float(np.linalg.norm(X - X_rec, ord="fro") / denom)


# Main Function
def evaluate_stability(func: Callable, X: np.ndarray, n_components: int = 2, noise_level: float = 0.01, random_state: int = 42, copy_data: bool = True, relative_noise: bool = True, **kwargs) -> Dict[str, Any]:
    """
    Evaluate stability of a PCA method under noise perturbation.

    Parameters
    ----------
    func : callable
        PCA method returning a dict.

    X : ndarray
        Input data matrix.

    n_components : int, default=2
        Number of principal components.

    noise_level : float, default=0.01
        Noise magnitude. If relative_noise=True, interpreted as a fraction
        of the standard deviation of X.

    random_state : int, default=42
        Random seed.

    copy_data : bool, default=True
        Whether to copy the input before PCA.

    relative_noise : bool, default=True
        If True, scale noise by std(X).

    Returns
    -------
    dict
        Stability metrics.
    """
    if n_components < 1:
        raise ValueError("n_components must be at least 1")
    if noise_level < 0:
        raise ValueError("noise_level must be non-negative")

    rng = np.random.default_rng(random_state)

    X_clean = X.copy() if copy_data else X

    sigma = noise_level * np.std(X_clean) if relative_noise else noise_level
    X_noisy = X_clean + sigma * rng.standard_normal(X.shape)

    res_clean = func(X_clean, n_components=n_components, **kwargs)
    res_noisy = func(X_noisy, n_components=n_components, **kwargs)

    if not isinstance(res_clean, dict) or not isinstance(res_noisy, dict):
        raise TypeError("PCA function must return a dict for stability evaluation.")

    results = {"projection_distance": None, "component_distance": None, "component_similarity": None, "explained_variance_distance": None, "reconstruction_error_delta": None,}

    Xp1 = _safe_get(res_clean, "X_proj")
    Xp2 = _safe_get(res_noisy, "X_proj")

    C1 = _safe_get(res_clean, "components")
    C2 = _safe_get(res_noisy, "components")

    evr1 = _safe_get(res_clean, "explained_variance_ratio")
    evr2 = _safe_get(res_noisy, "explained_variance_ratio")

    Xr1 = _safe_get(res_clean, "X_reconstructed")
    Xr2 = _safe_get(res_noisy, "X_reconstructed")

    if Xp1 is not None and Xp2 is not None:
        results["projection_distance"] = _projection_distance(Xp1, Xp2)

    if C1 is not None and C2 is not None:
        results["component_distance"] = _component_distance(C1, C2)
        results["component_similarity"] = _component_similarity(C1, C2)

    if evr1 is not None and evr2 is not None:
        results["explained_variance_distance"] = _explained_variance_distance(evr1, evr2)

    if Xr1 is not None and Xr2 is not None:
        err1 = _reconstruction_error(X_clean, Xr1)
        err2 = _reconstruction_error(X_noisy, Xr2)
        results["reconstruction_error_delta"] = abs(err1 - err2)

    return results


def print_stability_result(method_name: str, results: Dict[str, Any]) -> None:
    """
    Print formatted stability results.
    """
    print(f"\n{method_name}")
    print("-" * 50)

    for key, val in results.items():
        if val is None:
            print(f"{key:35}: N/A")
        else:
            print(f"{key:35}: {val:.6f}")

    print("-" * 50)