import numpy as np

def SVD(X, k=None):
    """
    PCA using SVD

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Input data matrix
    k : int, optional 
        Number of principal components to keep. If None, all components will be returned

    Returns
    -------
    U : ndarray
        Left Singular Vectors
    V : ndarray
        Right Singular Vectors
    S : ndarray
        Singular Values

    """


    #subtracting the mean of each feature
    X_centred = X - np.mean(X, axis=0)
    
    n_samples = X_centred.shape[0]

    U, S, Vt = np.linalg.svd(X_centred, full_matrices=False)
    V = Vt.T

    if k is not None:
        U = U[:,:k]
        S = S[:k]
        Vt = Vt[:k, :]
    return U, S, V

if __name__=="__main__":
    print("Running SVD as standalone script")
    np.random.seed(0)
    X = np.random.randn(200,10)

    U, S, V = SVD(X)

    print("U matrix shape:", U.shape)
    print("U matrix:", U)
    print("V matrix shape:", V.shape)
    print("V matrix:", V)
    print("S values:", S)