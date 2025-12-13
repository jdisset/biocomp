from functools import partial
from biocomp.logging_config import get_logger
import jax
import jax.numpy as jnp
import jaxkd as jk

logger = get_logger(__name__)


@partial(jax.jit, static_argnames=["k"])
def query_kdtree(queries, tree, k, distance_upper_bound):
    actual_k = min(len(tree.points), k)
    indices, distances = jk.query_neighbors(tree, queries, k=actual_k)
    too_far = distances > distance_upper_bound
    distances = jax.numpy.where(too_far, jnp.inf, distances)
    if k > len(tree.points):  # pad with inf distances
        ind_pad = jnp.zeros((queries.shape[0], k - len(tree.points)), dtype=jnp.int32)
        indices = jnp.concatenate([indices, ind_pad], axis=1, dtype=jnp.int32)  # type: ignore
        dist_pad = jnp.full((queries.shape[0], k - len(tree.points)), jnp.inf)
        distances = jnp.concatenate([distances, dist_pad], axis=1)  # type: ignore
    return distances, indices.astype(jnp.int32)


@partial(jax.jit, static_argnames=["k", "min_points", "normed_w"])
def get_gaussian_weighted_knn(
    x,
    tree,
    k: int = 500,  # number of neighbors to consider
    min_points: int = 20,  # minimum number of points to consider a neighborhood. fewer = nan
    radius: float = 0.1,
    sigma_in_radius: float = 3,  # sigma of the gaussian kernel in units of radius
    normed_w: bool = True,  # normalize weights
):
    """Get the k-nearest neighbors of x in the tree,
    and return their indices together with their weights (from a gaussian kernel)."""

    distances, indices = query_kdtree(x, tree, k=k, distance_upper_bound=radius)
    empty_neighbor_mask = distances == jnp.inf
    nb_points = (~empty_neighbor_mask).sum(axis=1)
    weights = jax.scipy.stats.norm.pdf(distances, 0, radius / sigma_in_radius)

    indices = jnp.where(empty_neighbor_mask, 0, indices)
    weights = jnp.where(empty_neighbor_mask, 0, weights)

    mask = nb_points < min_points
    weights = jnp.where(mask[:, None], jnp.nan, weights)

    if normed_w:
        weights = weights / jnp.nansum(weights, axis=1)[:, None]

    return indices, weights


@partial(jax.jit, static_argnames=["k", "min_points", "normed_w"])
def get_knn_mean_and_variance(x, y, iw=None, **kw):
    indices, weights = iw if iw is not None else get_gaussian_weighted_knn(x, **kw)

    y_neighbors = y[indices]
    weighted_mean = jnp.nansum(y_neighbors * weights[:, :, None], axis=1)
    squared_diff = (y_neighbors - weighted_mean[:, None, :]) ** 2
    variance = jnp.nansum(squared_diff * weights[:, :, None], axis=1)

    return weighted_mean, variance


@jax.jit
def weighted_quantile(data, weights, qu):
    ix = jnp.argsort(data)
    data = data[ix]
    weights = weights[ix]
    cdf = (jnp.cumsum(weights) - 0.5 * weights) / jnp.sum(weights)
    return jnp.interp(qu, cdf, data)


def get_knn_quantile(x, y, qu, iw=None, **kw):
    indices, weights = iw if iw is not None else get_gaussian_weighted_knn(x, **kw)
    return jax.vmap(weighted_quantile, in_axes=(0, 0, None))(y[indices], weights, qu)
