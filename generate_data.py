'''
Functions to generate synthetic data with a given 
number of data points and features for running PCA
Author: Suchet Sadekar
'''

import numpy as np
from scipy.stats import ortho_group

def get_correlated_data(n: int, k: int, noise: float = 0, random_state: int | None = None) -> tuple[np.ndarray[float], np.ndarray[float], np.ndarray[float]]:
    '''
    Generate a data matrix with correlated features.

    Parameters
    ----------
    n: int
        Number of data points
    k: int
        Dimensionality of data
    noise: float
        Standard Deviation of added Gaussian noise, default=0
    random_state: int or None
        Random seed (for reproducibility)

    Returns
    -------
    data: np.ndarray[floats] of shape (n, k)
        Data Matrix with 'n' rows and 'k' columns
    eigenvalues: np.ndarray[float] of shape (k, )
        Array with variance of each PC
    Q: np.ndarray[float] of shape (k, k) 
        Matrix with PCs as columns
    '''
    if random_state is not None:
        rng = np.random.default_rng(random_state)
    else:
        rng = np.random.default_rng()
    
    eigenvalues = 10 / (np.arange(1, k+1) ** 1.5) # Power-law decay for eigenvalues
    eigenvalues *= rng.lognormal(mean=0, sigma=0.1, size=k)
    
    Q = ortho_group.rvs(dim=k, random_state=rng) # Generate orthogonal matrix
    covariance = covariance = (Q * eigenvalues) @ Q.T @ Q.T # Get covariance matrix
    data = rng.multivariate_normal(mean = np.zeros(k), cov = covariance, size=n) # Sample from multivariate normal using covariance matrix
    if noise > 0:
        data += rng.normal(loc=0, scale=noise, size=data.shape) # Add gaussian noise
    data = data - data.mean(axis=0) # Center data
    return (data, eigenvalues, Q)

def get_uncorrelated_data(n: int, k: int, random_state: int | None = None) -> tuple[np.ndarray[float], np.ndarray[float], np.ndarray[float]]:
    '''
    Generate a data matrix with uncorrelated features.

    Parameters
    ----------
    n: int
        Number of data points
    k: int
        Dimensionality of data
    random_state: int
        Random seed (for reproducibility)

    Returns
    -------
    data: np.ndarray[floats] of shape (n, k)
        Data Matrix with 'n' rows and 'k' columns
    eigenvalues: np.ndarray[float] of shape (k, )
        Array with variance of each PC
    Q: np.ndarray[float] of shape (k, k) 
        Matrix with PCs as columns
    '''
    if random_state is not None:
        rng = np.random.default_rng(random_state)
    else:
        rng = np.random.default_rng()
    
    data = rng.normal(size=(n, k)) # Sample each entry independently from a normal distribution
    eigenvalues = np.ones(k)
    Q = np.eye(k)
    return (data, eigenvalues, Q)

def get_lowrank_data(n: int, k: int, r: int, noise: float = 0, random_state: int | None = None) -> tuple[np.ndarray[float], np.ndarray[float], np.ndarray[float]]:
    '''
    Generate a low-rank data matrix.
    
    Parameters:
    n: int
        Number of data points
    k: int
        Dimensionality of data
    r: int
        Rank of the data matrix
    noise: float
        Standard Deviation of added Gaussian noise, default=0
    random_state: int
        Random seed (for reproducibility)

    Returns
    -------
    data : np.ndarray[n, k] of floats
        Data matrix with 'n' rows and 'k' columns
    eigenvalues: np.ndarray[float] of shape (k, )
        Array with variance of each PC (1)
    Q: np.ndarray[float] of shape (k, k) 
        Matrix with PCs as columns
    '''
    assert 0 <= r <= k
    if random_state is not None:
        rng = np.random.default_rng(random_state)
    else:
        rng = np.random.default_rng()

    eigenvalues = np.zeros(k)
    eigenvalues[:r] = 10 / (np.arange(1, r+1) ** 1.5)
    Q = ortho_group.rvs(dim=k, random_state=rng)
    Z = rng.normal(size=(n, k))
    data = Z @ np.diag(np.sqrt(eigenvalues[:r])) @ Q[:, :r].T
    if noise > 0:
        data += rng.normal(scale=noise, size=data.shape)
    return (data, eigenvalues, Q)