import numpy as np
from biocomp.logging_config import get_logger
from scipy.stats import norm
from scipy.spatial import cKDTree


def knn_density(
    X: np.ndarray,
    k: int = 64,
    eps: float = 1e-12,
    tree: cKDTree | None = None,
) -> np.ndarray:
    """
    Compute density proxy using k-th nearest neighbor distance.
    Density ~ 1 / (d_k + eps)^D where D is dimensionality.
    Returns unnormalized density values suitable for importance sampling.
    """
    if tree is None:
        tree = cKDTree(X)
    d, _ = tree.query(X, k=k + 1)  # +1 because closest is self
    d_k = d[:, -1]  # distance to k-th neighbor
    dim = X.shape[1]
    return 1.0 / np.power(d_k + eps, dim)


def knn_density_chunked(
    X: np.ndarray,
    k: int = 64,
    eps: float = 1e-12,
    chunksize: int = 50000,
) -> np.ndarray:
    """
    Chunked version for large datasets to control memory usage.
    Builds tree once, queries in chunks.
    """
    tree = cKDTree(X)
    n = X.shape[0]
    dim = X.shape[1]
    result = np.empty(n, dtype=np.float64)
    for i in range(0, n, chunksize):
        end = min(i + chunksize, n)
        d, _ = tree.query(X[i:end], k=k + 1)
        d_k = d[:, -1]
        result[i:end] = 1.0 / np.power(d_k + eps, dim)
    return result


# def get_gaussian_weighted_knn(
#     x,
#     tree,
#     k: int = 500,  # number of neighbors to consider
#     min_points: int = 20,  # minimum number of points to consider a neighborhood. fewer = nan
#     radius: float = 0.1,
#     sigma_in_radius: float = 3,  # sigma of the gaussian kernel in units of radius
#     normed_w: bool = True,
# ):
#     """Get the k-nearest neighbors of x in the tree,
#     and return their indices together with their weights (from a gaussian kernel)."""
#
#     distances, indices = tree.query(x, k=k, distance_upper_bound=radius)
#     empty_neighbor_mask = distances == np.inf
#     nb_points = (~empty_neighbor_mask).sum(axis=1)
#     weights = norm.pdf(distances, loc=0, scale=radius / sigma_in_radius)
#     indices[empty_neighbor_mask] = 0
#     weights[empty_neighbor_mask] = 0
#     weights[nb_points < min_points, :] = np.nan
#
#     if normed_w:
#         row_sums = np.nansum(weights, axis=1)[:, None]
#         normalized_weights = np.full_like(weights, np.nan)
#         np.divide(weights, row_sums, out=normalized_weights, where=row_sums != 0)
#         return indices, normalized_weights
#
#     return indices, weights


def get_gaussian_weighted_knn(
    x,
    tree,
    k: int = 500,
    min_points: int = 20,
    radius: float = 0.1,  # fixed-kernel options (used when adaptive_sigma=False)
    sigma_in_radius: float = 3.0,  # radius ≈ sigma_in_radius * sigma
    adaptive_sigma: bool = False,  # per-query adaptive bandwidth (balloon estimator)
    # optional density reweighting
    densities: np.ndarray | None = None,  # len == n_observations
    density_power: float = 0.0,  # alpha in dens^-alpha; 0 disables, 1 = uniform
    density_floor: float | None = None,  # floor on densities before inversion
    density_cap: float | None = None,  # cap on densities before inversion
    normed_w: bool = True,
):
    """
    Get the k-nearest neighbors of x in the tree,
    and return their indices together with their weights (from a gaussian kernel).
    aka Nadaraya-Watson smoothing I think.

    If adaptive_sigma:
        - Query the k nearest neighbors (no radius cut).
        - Set sigma per query as (dist to k-th neighbor)/sigma_in_radius.
    Else:
        - Query up to k neighbors within 'radius' and set sigma = radius/sigma_in_radius.
        - Neighbors beyond radius get weight 0.

    If 'densities' is provided and density_power>0:
        multiply distance weights by densities[idx]**(-density_power),
        after applying optional floor/cap. Renormalize if normed_w=True.
    """
    eps = 1e-12

    if adaptive_sigma:
        distances, indices = tree.query(x, k=k)
        empty_neighbor_mask = ~np.isfinite(distances)
        nb_points = (~empty_neighbor_mask).sum(axis=1)

        # per-query sigma from the distance to the k-th neighbor
        # use the largest finite distance in the row if the k-th is inf
        kth = distances.copy()
        # with finite neighbors, take the maximum finite distance
        max_finite = np.where(np.isfinite(kth), kth, -np.inf).max(axis=1)
        sigma = (max_finite / sigma_in_radius).reshape(-1, 1) + eps
        # if a row has no finite neighbors, sigma will be eps (we'll NaN it below)
    else:
        # fixed-radius
        distances, indices = tree.query(x, k=k, distance_upper_bound=radius)
        empty_neighbor_mask = ~np.isfinite(distances)
        nb_points = (~empty_neighbor_mask).sum(axis=1)
        sigma = (radius / sigma_in_radius) + 0.0  # scalar

    Z = np.exp(-0.5 * (distances / sigma) ** 2)

    Z[empty_neighbor_mask] = 0.0
    indices = indices.copy()
    indices[empty_neighbor_mask] = 0

    if densities is not None and density_power > 0.0:  # density reweighting
        dens_nei = densities[indices]
        if density_floor is not None:
            dens_nei = np.maximum(dens_nei, density_floor)
        if density_cap is not None:
            dens_nei = np.minimum(dens_nei, density_cap)
        Z *= np.power(dens_nei + eps, -density_power)

    # too few neighbors -> NaN weights
    too_few = nb_points < min_points
    if np.any(too_few):
        Z[too_few, :] = np.nan

    if normed_w:
        row_sums = np.nansum(Z, axis=1, keepdims=True)
        W = np.full_like(Z, np.nan)
        np.divide(Z, row_sums, out=W, where=row_sums > 0)
        return indices, W

    return indices, Z


def get_knn_mean_and_variance(x, y, tree=None, iw=None, **kw):
    indices, weights = iw if iw is not None else get_gaussian_weighted_knn(x, tree=tree, **kw)

    y_neighbors = y[indices]  # (m, k, p)
    w = weights[..., None]  # (m, k, 1)
    weighted_mean = np.nansum(w * y_neighbors, axis=1)

    diff = y_neighbors - weighted_mean[:, None, :]
    var_num = np.nansum(w * (diff**2), axis=1)

    # DoF correction: divide by (1 - sum(w^2))
    w2sum = np.nansum((weights**2), axis=1, keepdims=True)
    denom = np.maximum(1.0 - w2sum, 1e-12)
    variance = var_num / denom

    # clean rows with all-NaN weights
    all_nan = np.all(np.isnan(weights), axis=1)
    weighted_mean[all_nan] = np.nan
    variance[all_nan] = np.nan
    return weighted_mean, variance
