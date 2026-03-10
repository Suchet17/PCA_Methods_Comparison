'''
Functions to generate synthetic data with a given 
number of data points and features for running PCA

Author: Suchet Sadekar
'''

import numpy as np
from scipy.stats import ortho_group

def get_correlated_data(n: int, k: int, noise: float = 0.0, random_state: int | None = None) -> dict:
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
    pc: np.ndarray[float] of shape (k, k) 
        Matrix with PCs as columns
    '''

    if random_state is not None:
        rng = np.random.default_rng(random_state)
    else:
        rng = np.random.default_rng()

    # Power-law decay for eigenvalues
    eigenvalues = 10 / (np.arange(1, k+1) ** 1.5)
    eigenvalues *= rng.lognormal(mean=0, sigma=0.1, size=k)

    # Generate orthogonal matrix
    pc = ortho_group.rvs(dim=k, random_state=rng)

    # Get covariance matrix
    covariance = (pc * eigenvalues) @ pc.T

    # Sample from multivariate normal using covariance matrix
    data = rng.multivariate_normal(mean = np.zeros(k), cov = covariance, size=n)

    # Add noise
    if noise > 0:
        data += rng.normal(loc=0, scale=noise, size=data.shape)

    # Center data
    data = data - data.mean(axis=0)
    return {'data': data, 'pcs': pc, 'variance': eigenvalues}

def get_uncorrelated_data(n: int, k: int, random_state: int | None = None) -> dict:
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
    pc: np.ndarray[float] of shape (k, k) 
        Matrix with PCs as columns
    '''

    if random_state is not None:
        rng = np.random.default_rng(random_state)
    else:
        rng = np.random.default_rng()

    # Sample each entry independently from a normal distribution
    data = rng.normal(size=(n, k))

    # True values for uncorrelated data
    eigenvalues = np.ones(k)
    pc = np.eye(k)

    return {'data': data, 'pcs': pc, 'variance': eigenvalues}

def get_lowrank_data(n: int, k: int, r: int, noise: float = 0.0, random_state: int | None = None) -> dict:
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
        Array with variance of each PC
    pc: np.ndarray[float] of shape (k, k) 
        Matrix with PCs as columns
    '''

    if random_state is not None:
        rng = np.random.default_rng(random_state)
    else:
        rng = np.random.default_rng()

    # Sample 'r' eigenvalues using power-law decay
    eigenvalues = np.zeros(k)
    eigenvalues[:r] = 10 / (np.arange(1, r+1) ** 1.5)

    # Generate orthogonal matrix
    pc = ortho_group.rvs(dim=k, random_state=rng)

    # Sample r-rank data
    z = rng.normal(size=(n, r))

    # transform to k-rank data
    data = (z * np.sqrt(eigenvalues[:r])) @ pc[:, :r].T

    # Add noise
    if noise > 0:
        data += rng.normal(scale=noise, size=data.shape)
    return {'data': data, 'pcs': pc, 'variance': eigenvalues}
