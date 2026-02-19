import numpy as np

def eigendecomposition(X, k=None):
    """
    PCA using eigenvalue decomposition of the covariance matrix.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features) - Input data matrix
    k : int, optional - Number of principal components to keep. 
        If None, all components will be returned

    Returns
    -------
    X_projected : ndarray
            Data projected onto principal component.
    eigenvalues : ndarray
            Sorted eigenvalues in descending order.
    eigenvectors : ndarray
            Corresponding eigenvectors.
    explained_variance_ratio : ndarray
            Variance explained by each selected component.
    """

    #subtracting the mean of each feature
    X_centred = X - np.mean(X, axis=0) 

    #covariance matric
    n_samples = X_centred.shape[0]
    covariance_matrix = (1 / (n_samples - 1 )) * X_centred.T @ X_centred


    #eigh is used as the covariance matrix is symmetric
    #eig will also compute the eigen values but it might be slower and less stable
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    #sorting is done in descending order as PCA requires the largest eigenvalue first
    sort_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues, eigenvectors = eigenvalues[sort_indices], eigenvectors[:, sort_indices]

    total_variance = np.sum(eigenvalues)

    if k is not None:
        eigenvalues = eigenvalues[:k]
        eigenvectors = eigenvectors[:, :k]

    X_projected = X_centred @ eigenvectors

    explained_variance_ratio = eigenvalues / total_variance

    return X_projected, eigenvalues, eigenvectors, explained_variance_ratio


if __name__ == "__main__":
    print("Running eigen_decompostion as standalone script")
    np.random.seed(0)
    X = np.random.randn(200, 10)

    X_projected, eigenvalues, eigenvectors, explained_variance_ratio = eigendecomposition(X, k=3)
    print("Projected shape:",X_projected.shape)
    print("Eigenvalues:", eigenvalues, "Eigenvectors:", eigenvectors)
    print("Variance_ratio:", explained_variance_ratio)

