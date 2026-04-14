# Comparative PCA Methods and Computational Complexity Analysis

**Collaborators**:\
Anshita Singh: anshita-singh-1043
Aurokrupa Sahoo: Aurokrupa\
Debagnik Das: Debug_Nuke\
Suchet Samir Sadekar: Suchet17

## Principal Component Analysis

Principal Component Analysis (PCA) is an unsupervised machine learning technique used for dimensionality reduction, data visualisation and noise filtering.
PCA linearly transforms a high-dimensional set of correlated features into a smaller set of uncorrelated variables (called the Principal Components), which maximise variance to retain maximum information.\
While PCA is widely used for dimensionality reduction, different implementations exhibit significant trade-offs depending on dataset size, dimensionality, and structure. This project presents a comparison of six Principal Component Analysis (PCA) implementations, analysing their runtime, memory usage, numerical stability, and scaling behaviour across controlled synthetic datasets.

### Features
- First-principles implementation of six PCA methods and verified against Scikit-learn reference implementations
- Shared benchmarking framework
- Synthetic data generation for correlated, non-correlated, low rank and non-linear data
- Analysis of the methods with different no. of observations(n) and no. of features(d) and their sensitivity to noise
  
### Structure

```
PCA_Methods_Comparison/
├── figs/                       # Plots for runtime, memory, scaling and stability analysis
│
├── generate_data.py            # Synthetic data generation 
├── generate_hyperspheres.py    # Non-linear dataset (for Kernel PCA)
│
├── eigen_decomposition.py      # PCA via eigendecomposition
├── svd.py                      # PCA via SVD
├── kernel_pca.py               # Kernel PCA implementation
├── Randomized_PCA.py           # Randomized PCA
├── incremental_pca.py          # Incremental PCA
├── sparse_pca.py               # Sparse PCA
│
├── pca_time.py                 # Runtime benchmarking
├── pca_memory.py               # Memory profiling
├── pca_scaling.py              # Scaling experiments
├── pca_stability.py            # Noise sensitivity analysis
├── pca_correctness.py          # Validation vs sklearn
│
├── kernel_pca_analyses.py      # Analysis script for Kernel PCA
├── *.txt                       # Output for experiments for different PCA methods
│
├── Project_Description.pdf
├── report.tex 
├── progress_report.pdf         
├── pca_final_report.pdf                 
│
├── README.md
└── .gitignore
```

## Complexity Summary
| Method	| Time Complexity |	Space Complexity |
|--------| ----------|---------|
| PCA by Eigendecomposition of covariance matrix |	O(nd² + d³) |	O(d²) |
| PCA by SVD	| O(min(n,d)² · max(n,d))	| O(nd) |
| Randomized PCA |	O(ndk) | O(nk + dk) |
| Incremental PCA	| O(bdk) | O(bd + kd) |
| Kernel PCA |	O(n³)	| O(n²) |
| Sparse PCA |  O(ndk⋅iter) | O(nk+dk)  |

## Insights
- PCA by Eigen decomposition of covariance matrix - simple and exact for small datasets
- PCA by SVD - stable and reliable for general use
- Randomized PCA - fastest for large datasets with small number of components
- Incremental PCA - suitable for large or streaming data
- Kernel PCA - effective for non-linear data but memory intensive
- Sparse PCA - useful when interpretability is important
