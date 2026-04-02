import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import eigh
from sklearn.decomposition import KernelPCA, PCA
from generate_hyperspheres import make_concentric_shells
import matplotlib.pyplot as plt

def k_pca(X, gamma, k=None):
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

  Author
  ------
  Anshita
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


if __name__ == "__main__":
  
  # IMPLEMENTING KPCA
  # generating data
  n = 1500
  X, y = make_concentric_shells(n_samples=n, radii=[0.3, 0.6, 0.9])

  # applying k_pca
  k = 2 
  gamma = 2
  noise = 0.03
  kpca_eigenval, kpca_projections, evr_kpca = k_pca(X, gamma, k)

  # applying sklearn.decomposition.PCA
  # getting all eigenvalues
  sk_kpca_all_components = KernelPCA(kernel='rbf',
                        gamma=gamma)
  sk_kpca_projections_all = sk_kpca_all_components.fit_transform(X)
  sk_eigenval_all = sk_kpca_all_components.eigenvalues_
  total_variance = np.sum(sk_eigenval_all)

  # getting k components only
  sk_kpca = KernelPCA(n_components=k,
                          kernel='rbf',
                          gamma=gamma)
  sk_kpca_projections = sk_kpca.fit_transform(X)
  sk_eigenvals = sk_kpca.eigenvalues_

  # explained variance ratio
  evr_sk_kpca = sk_eigenvals/total_variance

  # applying sklearn.decomposition.PCA
  pca = PCA(n_components=k)
  pca_projections = pca.fit_transform(X)
  evr_pca = pca.explained_variance_ratio_

  # VISUALISING 
  fig, ax = plt.subplots(1, 3, figsize=(15,5))

  # using manual k_pca method
  ax[0].scatter(kpca_projections[:,0], kpca_projections[:,1], c=y, cmap='viridis', alpha=0.6, edgecolors='k')
  ax[0].set_title(f"Manual implementation of KPCA (RBF kernel)\nPC1: {evr_kpca[0]:.2%}, PC2: {evr_kpca[1]:.2%}")

  # using sklearn.decomposition.KernelPCA
  ax[1].scatter(sk_kpca_projections[:,0], sk_kpca_projections[:,1], c=y, cmap='viridis', alpha=0.6, edgecolors='k')
  ax[1].set_title(f"Sklearn implementation of KPCA (RBF kernel)\nPC1: {evr_sk_kpca[0]:.2%}, PC2: {evr_sk_kpca[1]:.2%}")

  # using sklearn.decomposition.PCA
  image = ax[2].scatter(pca_projections[:,0], pca_projections[:,1], c=y, cmap='viridis', alpha=0.6, edgecolors='k')
  ax[2].set_title(f"Sklearn implementation of PCA\nPC1: {evr_pca[0]:.2%}, PC2: {evr_pca[1]:.2%}")

  for i in range(3):
    ax[i].set_xlabel("PC1")
    ax[i].set_ylabel("PC2")

  fig.colorbar(image, ax=ax[2], label='Sphere Index (Inner to Outer)')

fig.suptitle(f"Data: concentric n-dimensional hypersphers\nn: {n}, k: {k}, noise: {noise}, gamma: {gamma}")
plt.tight_layout()
plt.show()
