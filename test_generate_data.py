'''
Functions for testing the correctness of data generation
with respect to sklearn's output using pytest

Author: Suchet Sadekar
'''

import numpy as np
from sklearn.decomposition import PCA
from generate_data import get_correlated_data, get_uncorrelated_data, get_lowrank_data

def test_get_correlated_data():
    '''
    Check if generated correlated data's decomposition matches sklearn's decomposition
    '''
    pca = PCA()
    correlated = get_correlated_data(n = 100000, k = 10, noise = 0, random_state = 42)
    x = correlated['data']
    x -= x.mean(axis=0)
    pca.fit(x)

    # Ensure PCs are similar (within tolerance 0.1)
    print(np.abs(pca.components_ @ correlated['pcs']) - np.eye(x.shape[1]))
    assert np.allclose(np.abs(pca.components_ @ correlated['pcs']), np.eye(x.shape[1]), atol = 0.1)

    # Ensure explained variances are similar (within tolerance 0.1)
    print(pca.explained_variance_ - correlated['variance'])
    assert np.allclose(pca.explained_variance_, correlated['variance'], rtol = 0.1)

def test_get_uncorrelated_data():
    '''
    Check if generated uncorrelated data's decomposition matches sklearn's decomposition
    '''
    pca = PCA()
    uncorrelated = get_uncorrelated_data(n = 100000, k = 10, random_state = 42)
    x = uncorrelated['data']
    x -= x.mean(axis=0)
    pca.fit(x)

    # No need to ensure PCs are similar, not expected

    # Ensure explained variances are similar (within tolerance 0.1)
    print(pca.explained_variance_ - uncorrelated['variance'])
    assert np.allclose(pca.explained_variance_, uncorrelated['variance'], rtol = 0.1)

def test_get_lowrank_data():
    '''
    Check if generated lowrank data's decomposition matches sklearn's decomposition
    '''
    r = 3
    pca = PCA()
    lowrank = get_lowrank_data(n = 100000, k = 5, r = r, noise = 0, random_state = 42)
    x = lowrank['data']
    x -= x.mean(axis=0)
    pca.fit(x)

    # Ensure PCs are similar (within tolerance 0.1)
    print(np.abs(pca.components_[:r, :] @ lowrank['pcs'][:, :r]) - np.eye(r))
    assert np.allclose(np.abs(pca.components_[:r] @ lowrank['pcs'][:, :r]), np.eye(r), atol = 0.1)

    # Ensure lowrank explained variances are similar
    assert np.allclose(pca.explained_variance_[:r], lowrank['variance'][:r], rtol=0.1)

    # Ensure other explained variances are almost 0
    assert np.all(pca.explained_variance_[r:] < 1e-6)

if __name__ == '__main__':
    test_get_correlated_data()
    test_get_uncorrelated_data()
    test_get_lowrank_data()
