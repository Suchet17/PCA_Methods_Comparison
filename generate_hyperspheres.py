import numpy as np

def make_concentric_shells(n_samples=1500, n_dims=10, radii=None, noise=0.03):
    """
    Generates multiple concentric n-dimensional hyperspheres 
    (nested shells or n-balls of data points in a high-dimensional space).

    Parameters
    ----------
    n_samples : int, default=1500 
        Total number of data points 'n' to generate across all shells.
        The samples are divided equally among the number of shells provided
         in the `radii` list.
    n_dims : int, default=10
        Number of features (dimensions) 'd'.
    radii : list of float, default=[0.2, 0.4, 0.6, 0.8, 1.0]
        The radii of the concentric shells. Each value in the list creates 
         a new class of points at that distance from the origin.
    noise : float, default=0.03
        Standard deviation of Gaussian noise added to the coordinates.

    Returns
    -------
    X : ndarray of shape (n_samples, n_dims)
        The generated coordinates of the points in n-dimensional space.
    y : ndarray of shape (n_samples)
        The integer labels (0, 1, 2...) indicating which shell (radius) 
        each point belongs to.

    Notes
    -----
    The generation process follows three mathematical steps:
    1. Points are drawn from a standard multivariate normal distribution, 
       creating a spherically symmetric cloud.
    2. Each point's vector is normalized to unit length (projected to 
       the surface of a unit n-sphere).
    3. The unit vectors are scaled by the specified radius and perturbed 
       by random noise.

    Example
    -------
    >>> X, y = make_multi_hyperspheres(n_samples=500, n_dims=5, radii=[1, 5])
    >>> X.shape
    (500, 5)

    Author
    ------
    Anshita
    """

    if radii is None:
      radii = [0.2, 0.4, 0.6, 0.8, 1.0]

    # get number of shells and samples per shell
    n_classes = len(radii)
    samples_per_class = n_samples // n_classes
    
    # initialise X and y 
    X_list = []
    y_list = []
    
    for i, r in enumerate(radii):
        # generate Gaussian cloud
        X_shell = np.random.normal(size=(samples_per_class, n_dims))
        
        # project to unit sphere surface (normalization)
        X_shell /= np.linalg.norm(X_shell, axis=1)[:, np.newaxis]
        
        # scale by radius 
        X_shell *= r

        # add noise
        X_shell += np.random.normal(scale=noise, size=X_shell.shape)
        
        # append samples to X and y 
        X_list.append(X_shell)
        y_list.append(np.full(samples_per_class, i))
        
    return np.vstack(X_list), np.hstack(y_list)
