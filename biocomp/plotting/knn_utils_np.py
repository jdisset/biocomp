import os
import numpy as np
from scipy.spatial import cKDTree


try:
    KNN_WORKERS = int(os.environ.get("BIOCOMP_KNN_WORKERS", "-1"))
except ValueError:
    KNN_WORKERS = -1

try:
    KNN_MEAN_CHUNK_SIZE = int(os.environ.get("BIOCOMP_KNN_MEAN_CHUNK_SIZE", "2500"))
except ValueError:
    KNN_MEAN_CHUNK_SIZE = 2500

try:
    from pykdtree.kdtree import KDTree as _PKDTree
    if KNN_WORKERS == 1:
        os.environ.setdefault("OMP_NUM_THREADS", "1")
except ImportError:
    _PKDTree = None


def make_tree(x: np.ndarray):
    if _PKDTree is not None:
        return _PKDTree(np.ascontiguousarray(x, dtype=np.float64))
    return cKDTree(x, leafsize=32)


def _query(tree, x, **kw):
    if _PKDTree is not None and isinstance(tree, _PKDTree):
        kw.pop("workers", None)
    return tree.query(x, **kw)


def knn_density(
    X: np.ndarray,
    k: int = 64,
    eps: float = 1e-12,
    tree=None,
) -> np.ndarray:
    """
    Compute density proxy using k-th nearest neighbor distance.
    Density ~ 1 / (d_k + eps)^D where D is dimensionality.
    Returns unnormalized density values suitable for importance sampling.
    """
    if tree is None:
        tree = make_tree(X)
    d, _ = _query(tree, X, k=k + 1)
    d_k = d[:, -1]
    dim = X.shape[1]
    return 1.0 / np.power(d_k + eps, dim)


def knn_density_chunked(
    X: np.ndarray,
    k: int = 64,
    eps: float = 1e-12,
    chunksize: int = 50000,
    tree=None,
) -> np.ndarray:
    """
    Chunked version for large datasets to control memory usage.
    Builds tree once, queries in chunks.
    """
    if tree is None:
        tree = make_tree(X)
    n = X.shape[0]
    dim = X.shape[1]
    result = np.empty(n, dtype=np.float64)
    for i in range(0, n, chunksize):
        end = min(i + chunksize, n)
        d, _ = _query(tree, X[i:end], k=k + 1, workers=KNN_WORKERS)
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
    max_radius: float | None = None,  # hard cutoff for adaptive_sigma
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
        - If max_radius is set, zero out weights for neighbors beyond max_radius.
    Else:
        - Query up to k neighbors within 'radius' and set sigma = radius/sigma_in_radius.
        - Neighbors beyond radius get weight 0.

    If 'densities' is provided and density_power>0:
        multiply distance weights by densities[idx]**(-density_power),
        after applying optional floor/cap. Renormalize if normed_w=True.
    """
    eps = 1e-12

    if adaptive_sigma:
        distances, indices = _query(tree, x, k=k, workers=KNN_WORKERS)
        finite_mask = np.isfinite(distances)

        # per-query sigma from the largest finite distance in each row
        max_finite = np.where(finite_mask, distances, -np.inf).max(axis=1)
        sigma = (max_finite / sigma_in_radius).reshape(-1, 1) + eps

        if max_radius is not None:
            valid_mask = finite_mask & (distances <= max_radius)
        else:
            valid_mask = finite_mask

        nb_points = valid_mask.sum(axis=1)
    else:
        distances, indices = _query(tree, x, k=k, distance_upper_bound=radius, workers=KNN_WORKERS)
        valid_mask = np.isfinite(distances)
        nb_points = valid_mask.sum(axis=1)
        sigma = (radius / sigma_in_radius) + 0.0  # scalar

    too_few = nb_points < min_points
    enough = ~too_few
    invalid_mask = ~valid_mask
    if invalid_mask.any():
        indices = indices.copy()
        indices[invalid_mask] = 0

    if too_few.any() and not enough.any():
        return indices, np.full_like(distances, np.nan)

    if too_few.any():
        d_v = distances[enough]
        v_v = valid_mask[enough]
        inv_sigma_v = 1.0 / (sigma[enough] if not np.isscalar(sigma) else sigma)
        Z_v = d_v * inv_sigma_v
        Z_v *= Z_v
        Z_v *= -0.5
        np.exp(Z_v, out=Z_v)
        Z_v[~v_v] = 0.0

        if densities is not None and density_power > 0.0:
            dens_nei = densities[indices[enough]]
            if density_floor is not None:
                dens_nei = np.maximum(dens_nei, density_floor)
            if density_cap is not None:
                dens_nei = np.minimum(dens_nei, density_cap)
            Z_v *= np.power(dens_nei + eps, -density_power)

        if normed_w:
            row_sums = Z_v.sum(axis=1, keepdims=True)
            W_v = np.full_like(Z_v, np.nan)
            np.divide(Z_v, row_sums, out=W_v, where=row_sums > 0)
            W = np.full_like(distances, np.nan)
            W[enough] = W_v
            return indices, W
        Z = np.full_like(distances, np.nan)
        Z[enough] = Z_v
        return indices, Z

    inv_sigma = 1.0 / sigma
    Z = distances * inv_sigma
    Z *= Z
    Z *= -0.5
    np.exp(Z, out=Z)
    Z[invalid_mask] = 0.0

    if densities is not None and density_power > 0.0:
        dens_nei = densities[indices]
        if density_floor is not None:
            dens_nei = np.maximum(dens_nei, density_floor)
        if density_cap is not None:
            dens_nei = np.minimum(dens_nei, density_cap)
        Z *= np.power(dens_nei + eps, -density_power)

    if normed_w:
        row_sums = Z.sum(axis=1, keepdims=True)
        W = np.full_like(Z, np.nan)
        np.divide(Z, row_sums, out=W, where=row_sums > 0)
        return indices, W

    return indices, Z


def get_knn_mean_and_variance(x, y, tree=None, iw=None, compute_variance=True, **kw):
    indices, weights = iw if iw is not None else get_gaussian_weighted_knn(x, tree=tree, **kw)

    n_grid = indices.shape[0]
    valid_rows = np.isfinite(weights[:, 0])
    n_outs = y.shape[1] if y.ndim > 1 else 1
    all_valid = valid_rows.all()

    if all_valid:
        ind_v, w_v = indices, weights
    else:
        ind_v, w_v = indices[valid_rows], weights[valid_rows]

    y_neighbors = y[ind_v]
    w = w_v[..., None]
    wy = w * y_neighbors
    mean_v = wy.sum(axis=1)
    if compute_variance:
        second_moment = (wy * y_neighbors).sum(axis=1)
        w2sum = (w_v * w_v).sum(axis=1, keepdims=True)
        var_v = (second_moment - mean_v * mean_v) / np.maximum(1.0 - w2sum, 1e-12)
    else:
        var_v = None

    if all_valid:
        all_nan = np.all(np.isnan(weights), axis=1)
        if all_nan.any():
            mean_v[all_nan] = np.nan
            if var_v is not None:
                var_v[all_nan] = np.nan
        return mean_v, var_v

    weighted_mean = np.full((n_grid, n_outs), np.nan, dtype=mean_v.dtype)
    weighted_mean[valid_rows] = mean_v
    variance = None
    if var_v is not None:
        variance = np.full((n_grid, n_outs), np.nan, dtype=var_v.dtype)
        variance[valid_rows] = var_v
    return weighted_mean, variance


def _knn_mean_from_indices_weights(indices, weights, y):
    n_grid = indices.shape[0]
    valid_rows = np.isfinite(weights[:, 0])
    n_outs = y.shape[1] if y.ndim > 1 else 1

    if valid_rows.all():
        row_sums = weights.sum(axis=1, keepdims=True)
        np.divide(weights, row_sums, out=weights, where=row_sums > 0)
        y_neighbors = y[indices]
        if y.ndim == 1:
            y_neighbors *= weights
            return y_neighbors.sum(axis=1, keepdims=True)
        y_neighbors *= weights[..., None]
        return y_neighbors.sum(axis=1)

    if not valid_rows.any():
        return np.full((n_grid, n_outs), np.nan, dtype=y.dtype)

    ind_v = indices[valid_rows]
    w_v = weights[valid_rows]
    row_sums = w_v.sum(axis=1, keepdims=True)
    np.divide(w_v, row_sums, out=w_v, where=row_sums > 0)
    y_neighbors = y[ind_v]
    if y.ndim == 1:
        y_neighbors *= w_v
        mean_v = y_neighbors.sum(axis=1, keepdims=True)
    else:
        y_neighbors *= w_v[..., None]
        mean_v = y_neighbors.sum(axis=1)

    weighted_mean = np.full((n_grid, n_outs), np.nan, dtype=mean_v.dtype)
    weighted_mean[valid_rows] = mean_v
    return weighted_mean


def get_knn_mean_only(x, y, tree=None, iw=None, **kw):
    if iw is not None:
        return _knn_mean_from_indices_weights(iw[0], iw[1], y)

    chunk_size = KNN_MEAN_CHUNK_SIZE
    if chunk_size > 0 and x.shape[0] > chunk_size:
        chunks = []
        for start in range(0, x.shape[0], chunk_size):
            stop = min(start + chunk_size, x.shape[0])
            indices, weights = get_gaussian_weighted_knn(
                x[start:stop],
                tree=tree,
                normed_w=False,
                **kw,
            )
            chunks.append(_knn_mean_from_indices_weights(indices, weights, y))
        return np.concatenate(chunks, axis=0)

    indices, weights = get_gaussian_weighted_knn(x, tree=tree, normed_w=False, **kw)
    return _knn_mean_from_indices_weights(indices, weights, y)
