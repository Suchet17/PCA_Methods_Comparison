import numpy as np
import matplotlib.pyplot as plt
import time


# STANDARDIZATION (SAFE)

def standardize(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std[std == 0] = 1   # avoid division by zero
    return (X - mean) / std



# NORMAL PCA (USING SVD)

def normal_pca(X, k):
    X_centered = X - np.mean(X, axis=0)

    # SVD
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

    # Top k components
    U_k = U[:, :k]
    S_k = S[:k]
    Vt_k = Vt[:k, :]

    # Explained variance ratio
    variance = S_k**2
    total_variance = np.sum(S**2)
    evr = variance / total_variance

    return U_k, S_k, Vt_k, evr



# RANDOMIZED PCA

def randomized_pca(X, k, oversample=5):
    X_centered = X - np.mean(X, axis=0)
    n, d = X_centered.shape

    # Step 1: Random projection
    P = np.random.randn(d, k + oversample)
    Z = X_centered @ P

    # Step 2: QR decomposition
    Q, _ = np.linalg.qr(Z)

    # Step 3: Small matrix
    B = Q.T @ X_centered

    # Step 4: SVD on small matrix
    U_hat, S, Vt = np.linalg.svd(B, full_matrices=False)

    # Step 5: Recover U
    U = Q @ U_hat

    # Top k components
    U_k = U[:, :k]
    S_k = S[:k]
    Vt_k = Vt[:k, :]

    # Use original data for total variance
    _, S_full, _ = np.linalg.svd(X_centered, full_matrices=False)
    variance = S_k**2
    total_variance = np.sum(S_full**2)
    evr = variance / total_variance

    return U_k, S_k, Vt_k, evr



# GENERATE DATA

np.random.seed(42)

n = 1000   # samples
d = 50     # features

k = int(input("Enter number of principal components: "))

# Random data
X = np.random.randn(n, d)

# Add some correlation (moderate, not extreme)
X[:, 1] = X[:, 0] * 2 + np.random.randn(n) * 0.1
X[:, 2] = X[:, 0] * -1.5 + np.random.randn(n) * 0.1

# Standardize
X_std = standardize(X)



# RUN RANDOMIZED PCA

start = time.time()
X_rpca, S_rpca, Vt_rpca, evr_rpca = randomized_pca(X_std, k)
rpca_time = time.time() - start


# RUN NORMAL PCA
start = time.time()
X_pca, S_pca, Vt_pca, evr_pca = normal_pca(X_std, k)
pca_time = time.time() - start


# PRINT RESULTS

print("\nPrincipal Components (Randomized PCA):\n", Vt_rpca)
print("\nPrincipal Components (Normal PCA):\n", Vt_pca)

print("\nRandomized PCA Singular Values:", S_rpca)
print("Randomized PCA EVR:", evr_rpca)

print("\nNormal PCA Singular Values:", S_pca)
print("Normal PCA EVR:", evr_pca)

print(f"\nRandomized PCA Time: {rpca_time:.5f} sec")
print(f"Normal PCA Time: {pca_time:.5f} sec")

# Difference check (important for validation)
print("\nDifference in EVR:", np.abs(evr_pca - evr_rpca))


# VISUALIZATION

fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# Randomized PCA
ax[0].scatter(X_rpca[:, 0], X_rpca[:, 1], alpha=0.5, edgecolor='k')
ax[0].set_title(f"Randomized PCA\nPC1: {evr_rpca[0]:.2%}, PC2: {evr_rpca[1]:.2%}")
ax[0].set_xlabel("PC1")
ax[0].set_ylabel("PC2")

# Normal PCA
ax[1].scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.5, edgecolor='k')
ax[1].set_title(f"Normal PCA\nPC1: {evr_pca[0]:.2%}, PC2: {evr_pca[1]:.2%}")
ax[1].set_xlabel("PC1")
ax[1].set_ylabel("PC2")

plt.suptitle("Randomized PCA vs Normal PCA")
plt.tight_layout()
plt.show()
