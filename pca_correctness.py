"""
Generic correctness evaluation for PCA methods.

Required function signature
---------------------------
func(X, n_components=..., **kwargs) -> dict

Preferred returned dictionary keys
----------------------------------
{   "X_proj": ndarray,                     # projected data
    "components": ndarray or None,         # principal directions in feature space if available
    "explained_variance_ratio": ndarray or None,
    "X_reconstructed": ndarray or None,
    "model": object
}
Notes
-----
1. Kernel PCA may not provide linear components
   in the original feature space. In that case, component-based metrics
   are skipped automatically.
2. Sparse PCA may return components in a different order, so component
   matching is supported.
"""

import numpy as np

def _safe_get(result, key):
    if result is None:
        return None
    return result.get(key, None)


def _normalize_rows(A):
    """
    Normalize each row vector to unit norm.
    """
    A = np.asarray(A, dtype=float)
    norms = np.linalg.norm(A, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return A / norms

def _align_signs_rowwise(A, B):
    """
    Align signs of rows of A to match B.
    """
    A = A.copy()
    n = min(A.shape[0], B.shape[0])
    for i in range(n):
        if np.dot(A[i], B[i]) < 0:
            A[i] *= -1
    return A

def _align_projection_signs(A, B):
    """
    Align signs of projected coordinates column-wise.
    """
    A = A.copy()
    n = min(A.shape[1], B.shape[1])
    for i in range(n):
        if np.dot(A[:, i], B[:, i]) < 0:
            A[:, i] *= -1
    return A

def _match_components_by_similarity(A, B):
    """
    Reorder rows of A to best match rows of B using absolute cosine similarity.
    Useful for sparse PCA or other methods where component ordering may vary.
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    n = min(A.shape[0], B.shape[0])
    A = A[:n]
    B = B[:n]
    A_norm = _normalize_rows(A)
    B_norm = _normalize_rows(B)
    similarity = np.abs(A_norm @ B_norm.T)
    matched_rows = []
    used_rows = set()
    for j in range(n):
        best_i = None
        best_val = -np.inf
        for i in range(n):
            if i in used_rows:
                continue
            if similarity[i, j] > best_val:
                best_val = similarity[i, j]
                best_i = i
        used_rows.add(best_i)
        matched_rows.append(A[best_i])
    matched_A = np.array(matched_rows)
    matched_A = _align_signs_rowwise(matched_A, B)
    return matched_A, B

def _projection_distance(A, B):
    """
    Relative distance between projected coordinates.
    Lower is better.
    """
    n = min(A.shape[1], B.shape[1])
    A = A[:, :n]
    B = B[:, :n]
    A = _align_projection_signs(A, B)
    denom = np.linalg.norm(B, ord="fro")
    if denom == 0:
        return float(np.linalg.norm(A - B, ord="fro"))
    return float(np.linalg.norm(A - B, ord="fro") / denom)

def _component_distance(A, B, match_components=True):
    """
    Relative distance between component matrices.
    Lower is better.
    """
    if match_components:
        A, B = _match_components_by_similarity(A, B)
    else:
        n = min(A.shape[0], B.shape[0])
        A = _align_signs_rowwise(A[:n], B[:n])
        B = B[:n]
    denom = np.linalg.norm(B, ord="fro")
    if denom == 0:
        return float(np.linalg.norm(A - B, ord="fro"))
    return float(np.linalg.norm(A - B, ord="fro") / denom)

def _subspace_similarity(A, B):
    """
    Mean absolute cosine similarity between matched components.
    Higher is better. Maximum = 1.
    """
    A, B = _match_components_by_similarity(A, B)
    sims = []
    for a, b in zip(A, B):
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na == 0 or nb == 0:
            sims.append(0.0)
        else:
            sims.append(abs(np.dot(a, b) / (na * nb)))
    return float(np.mean(sims)) if sims else None


def _subspace_overlap(A, B):
    """
    Similarity between the subspaces spanned by A and B.
    Higher is better. Maximum = 1.
    """
    n = min(A.shape[0], B.shape[0])
    A = A[:n]
    B = B[:n]
    QA, _ = np.linalg.qr(A.T)
    QB, _ = np.linalg.qr(B.T)
    M = QA.T @ QB
    s = np.linalg.svd(M, compute_uv=False)
    return float(np.mean(np.clip(s, 0.0, 1.0))) if len(s) > 0 else None

def _explained_variance_distance(A, B):
    """
    Relative distance between explained variance ratio vectors.
    Lower is better.
    """
    n = min(len(A), len(B))
    A = np.asarray(A[:n], dtype=float)
    B = np.asarray(B[:n], dtype=float)
    denom = np.linalg.norm(B)
    if denom == 0:
        return float(np.linalg.norm(A - B))
    return float(np.linalg.norm(A - B) / denom)


def _reconstruction_error(X, X_rec):
    """
    Normalized reconstruction error.
    Lower is better.
    """
    denom = np.linalg.norm(X, ord="fro")
    if denom == 0:
        return 0.0

    return float(np.linalg.norm(X - X_rec, ord="fro") / denom)

def evaluate_correctness(test_func, reference_func, X, n_components=2, test_kwargs=None, reference_kwargs=None, copy_data=True, match_components=True,):
    """
    Evaluate the correctness of a PCA-like method against a reference method.
    This function is generic and can be used for standard, sparse,
    randomized, incremental, and kernel PCA.

    Parameters
    ----------
    test_func : callable
        PCA method being tested.
    reference_func : callable
        Reference PCA method.
    X : ndarray
        Input data matrix.
    n_components : int, default=2
        Number of retained components.
    test_kwargs : dict or None, default=None
        Extra keyword arguments for test_func.
    reference_kwargs : dict or None, default=None
        Extra keyword arguments for reference_func.
    copy_data : bool, default=True
        Whether to pass copies of X.
    match_components : bool, default=True
        Whether to reorder test components to best match reference components.

    Returns
    -------
    dict
        Correctness metrics. Metrics that are not applicable are returned as None.
    """
    test_kwargs = test_kwargs or {}
    reference_kwargs = reference_kwargs or {}
    X_test = X.copy() if copy_data else X
    X_ref = X.copy() if copy_data else X
    test_result = test_func(X_test, n_components=n_components, **test_kwargs)
    ref_result = reference_func(X_ref, n_components=n_components, **reference_kwargs)
    results = {
        "projection_distance": None,
        "component_distance": None,
        "subspace_similarity": None,
        "subspace_overlap": None,
        "explained_variance_distance": None,
        "test_reconstruction_error": None,
        "reference_reconstruction_error": None,
        "reconstruction_error_gap": None,}

    Xp_test = _safe_get(test_result, "X_proj")
    Xp_ref = _safe_get(ref_result, "X_proj")

    C_test = _safe_get(test_result, "components")
    C_ref = _safe_get(ref_result, "components")

    evr_test = _safe_get(test_result, "explained_variance_ratio")
    evr_ref = _safe_get(ref_result, "explained_variance_ratio")

    Xr_test = _safe_get(test_result, "X_reconstructed")
    Xr_ref = _safe_get(ref_result, "X_reconstructed")

    # Projection comparison: valid for almost all PCA-like methods
    if Xp_test is not None and Xp_ref is not None:
        results["projection_distance"] = _projection_distance(Xp_test, Xp_ref)

    # Component/subspace comparison: only if both methods expose components
    if C_test is not None and C_ref is not None:
        results["component_distance"] = _component_distance(
            C_test, C_ref, match_components=match_components)
        results["subspace_similarity"] = _subspace_similarity(C_test, C_ref)
        results["subspace_overlap"] = _subspace_overlap(C_test, C_ref)

    # Explained variance comparison: only if available
    if evr_test is not None and evr_ref is not None:
        results["explained_variance_distance"] = _explained_variance_distance(
            evr_test, evr_ref
        )

    # Reconstruction comparison: only if available
    if Xr_test is not None:
        results["test_reconstruction_error"] = _reconstruction_error(X, Xr_test)

    if Xr_ref is not None:
        results["reference_reconstruction_error"] = _reconstruction_error(X, Xr_ref)

    if (results["test_reconstruction_error"] is not None
        and results["reference_reconstruction_error"] is not None):
        results["reconstruction_error_gap"] = abs(
            results["test_reconstruction_error"]
            - results["reference_reconstruction_error"]
        )
    return results


def print_correctness_result(method_name, results):
    """
    Print correctness metrics in a clean format.
    """
    print(f"\n{method_name}")
    print("-" * 80)
    for key, value in results.items():
        if value is None:
            print(f"{key:35}: N/A")
        else:
            print(f"{key:35}: {value}")
    print("-" * 80)