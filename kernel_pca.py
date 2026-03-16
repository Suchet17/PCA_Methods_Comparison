import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import eigh
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import KernelPCA, PCA
from generate_data import get_correlated_data, get_uncorrelated_data, get_lowrank_data
import matplotlib.pyplot as plt

def kpca(X, gamma, k=None):
  """
  Kernel PCA implementation (from first principles), using RBF kernel.

  Parameters
  ----------
  X : ndarray[float] of shape (N, d)
    The input data matrix containing raw features,
    for N points and d features
  gamma : float
    Hyperparameter for the Radial Basis Function (RBF) kernel;
    defines the 'spread' of the kernel
  k : int, optional
    Number of principle components to return

  Returns
  -------
  alphas : ndarray of shape (N, k)
    The top eigenvectors of the centered kernel matrix;
    represent data projected onto the new principle component axes
  lambdas : list
    The eigenvalues corresponding to the selected principle components
  """

  ## computing and centering kernel matrix K

  # calculating pairwise squared euclidean distances
  sq_dists = squareform(pdist(X, 'sqeuclidean'))  #pdist gives 1d array, squareform to make symmetric matrix

  # computing K
  K = np.exp(-gamma * sq_dists)

  # centering K to get K_c
  N = K.shape[0]  #no. of points
  one_N = np.ones((N,N)) / N  #N x N matrix with entries 1/N
  K_c = K - np.dot(one_N, K) - np.dot(K, one_N) + np.dot(one_N, np.dot(K, one_N)) 

  ## eigenvalue decomposition

  # getting all eigenvalues and eigenvectors (in ascending order)
  eigenvalues, eigenvectors = eigh(K_c)

  # sorting in descending order
  idx = np.argsort(eigenvalues)[::-1]
  eigenvalues = eigenvalues[idx]
  eigenvectors = eigenvectors[:, idx]

  # removing negative values (numerical noise)
  eigenvalues = np.maximum(eigenvalues, 0)

  total_variance = np.sum(eigenvalues)

  # extracting the first k eigenvectors and corresponding eigenvalues, if k is not None
  if k is not None:
    lambdas = eigenvalues[:k]
    alphas = eigenvectors[:, :k] * np.sqrt(lambdas) #sclaing to make projection comaprable with sklearn
  else:
    lambdas = eigenvalues
    alphas = eigenvectors * np.sqrt(lambdas)

  explained_variance_ratio = lambdas/total_variance

  return lambdas, alphas, explained_variance_ratio


# IMPLEMENTATION
n = 1000
k = 100
noise = 0.05
gamma = 0.00001

# generating correlated data
corr_data = get_correlated_data(n=n, k=k, noise=noise, random_state=42)
X = corr_data['data']

# standardising the data 
X_std = StandardScaler().fit_transform(X)

# applying kpca
lambdas, alphas, evr = kpca(X_std, gamma, k)

## for comparison with sklearn methods

# PCA: sklearn.decomposition.PCA
pca = PCA(n_components=2)
X_pca_sk = pca.fit_transform(X_std)
evr_pca = pca.explained_variance_ratio_

# KPCA: sklearn.decomposition.KernelPCA
# getting all eigenvalues 
transformer = KernelPCA(kernel='rbf', 
                        gamma=gamma)
X_std_transformed = transformer.fit_transform(X_std)
eigenvals = transformer.eigenvalues_
total_variance = np.sum(eigenvals)

# getting k components only 
transformer_2 = KernelPCA(n_components=k, 
                          kernel='rbf', 
                          gamma=gamma)
X_transformed_2 = transformer_2.fit_transform(X_std)
eigenvals_2 = transformer_2.eigenvalues_

# explained variance ratio 
evr_kpca = eigenvals_2/total_variance


# VISUALISING 
fig, ax = plt.subplots(1, 3, figsize=(18,5))

# using sklearn.decomposition.PCA
ax[0].scatter(X_pca_sk[:, 0], X_pca_sk[:, 1], alpha=0.5, color='DarkSlateBlue', edgecolor='k')
ax[0].set_title(f"Sklearn PCA\nPC1: {evr_pca[0]:.2%}, PC2: {evr_pca[1]:.2%}")
ax[0].set_xlabel("PC1")
ax[0].set_ylabel("PC2")

# using manual kpca method
ax[1].scatter(alphas[:, 0], alphas[:, 1], c='DarkSlateBlue', alpha=0.5, edgecolor='k')
ax[1].set_title(f"Manual KPCA\nPC1: {evr[0]:.2%}, PC2: {evr[1]:.2%}")
ax[1].set_xlabel("PC1")
ax[1].set_ylabel("PC2")

# using sklearn.decomposition.KernelPCA
ax[2].scatter(X_transformed_2[:, 0], X_transformed_2[:, 1], c='DarkSlateBlue', alpha=0.5, edgecolor='k')
ax[2].set_title(f"Sklearn KernelPCA\nPC1: {evr_kpca[0]:.2%}, PC2: {evr_kpca[1]:.2%}")
ax[2].set_xlabel("PC1")
ax[2].set_ylabel("PC2")

fig.suptitle(f"Implementation for correlated data\nn: {n}, k: {k}, noise: {noise}, gamma: {gamma}")
plt.tight_layout()
plt.show()
