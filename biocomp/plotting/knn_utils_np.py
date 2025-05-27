import numpy as np
from biocomp.logging_config import get_logger
from scipy.spatial import KDTree
from scipy.stats import norm

logger = get_logger(__name__)


def get_gaussian_weighted_knn(
    x,
    tree,
    k: int = 500,  # number of neighbors to consider
    min_points: int = 20,  # minimum number of points to consider a neighborhood. fewer = nan
    radius: float = 0.1,
    sigma_in_radius: float = 3,  # sigma of the gaussian kernel in units of radius
    normed_w: bool = True,
):
    """Get the k-nearest neighbors of x in the tree,
    and return their indices together with their weights (from a gaussian kernel)."""

    distances, indices = tree.query(x, k=k, distance_upper_bound=radius)
    empty_neighbor_mask = distances == np.inf
    nb_points = (~empty_neighbor_mask).sum(axis=1)
    weights = norm.pdf(distances, loc=0, scale=radius / sigma_in_radius)
    indices[empty_neighbor_mask] = 0
    weights[empty_neighbor_mask] = 0
    weights[nb_points < min_points, :] = np.nan

    if normed_w:
        row_sums = np.nansum(weights, axis=1)[:, None]
        normalized_weights = np.full_like(weights, np.nan)
        np.divide(weights, row_sums, out=normalized_weights, where=row_sums != 0)
        return indices, normalized_weights

    return indices, weights


def get_knn_mean_and_variance(x, y, tree=None, iw=None, **kw):
    indices, weights = iw if iw is not None else get_gaussian_weighted_knn(x, tree=tree, **kw)

    y_neighbors = y[indices]
    weighted_mean = np.nansum(y_neighbors * weights[:, :, None], axis=1)
    squared_diff = (y_neighbors - weighted_mean[:, None, :]) ** 2
    variance = np.nansum(squared_diff * weights[:, :, None], axis=1)

    return weighted_mean, variance
