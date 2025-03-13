# {{{                          --     imports     --
# ···············································································
from functools import partial
import numpy as np
from biocomp import utils as ut
from typing import Literal
from biocomp.logging_config import get_logger
import jax
import jax.numpy as jnp
from jax import vmap

logger = get_logger(__name__)

##────────────────────────────────────────────────────────────────────────────}}}

configurable = ut.configurable_decorator("biocomp.plotting")


class SpatialQueryGrid:
    def __init__(
        self,
        data: np.ndarray,
        resolution: int = 200,
    ):
        """Initialize the spatial grid structure"""
        self.__data = data
        self.__resolution = resolution
        # save data to /tmp/datadump.npy
        # np.save("/tmp/datadump.npy", data)

        self.make_grid()
        self.make_query_fn()

    def make_grid(self, bin_capacity: int | Literal["auto"] = "auto") -> None:
        """Create the spatial grid structure. Specifying bin_capacity instead 'auto' allows this method to be jitted."""

        import jax
        import jax.numpy as jnp
        from jax import vmap

        @jax.jit
        def get_bin(point) -> int:
            nd_pos = self.get_bin_nd(point)
            return jnp.ravel_multi_index(nd_pos, self.__grid_shape, mode="clip")  # type: ignore

        self.__lower = jnp.min(self.__data, axis=0)
        self.__upper = jnp.max(self.__data, axis=0)

        self.__binsize = jnp.min((self.__upper - self.__lower) / self.__resolution)
        self.__grid_shape = tuple(
            np.ceil((self.__upper - self.__lower) / self.__binsize).astype(int)
        )

        n_points = len(self.__data)

        point_indices = jnp.arange(n_points)
        point_positions = vmap(get_bin)(self.__data)
        bin_counts = jnp.bincount(point_positions)
        sorted_point_positions = point_indices[jnp.argsort(point_positions)]

        if bin_capacity == "auto":
            bin_capacity = int(jnp.max(bin_counts))
        else:
            bin_capacity = bin_capacity

        last_elt = jnp.full((1, bin_capacity), -1)  # just so that grid[-1] returns an "empty" bin

        @jax.jit
        def make_grid(bin_counts):
            csum = jnp.cumsum(bin_counts)
            start_idx = jnp.concatenate((jnp.zeros(1), csum[:-1])).astype(int)

            @vmap
            def impl(start_idx, end_idx):
                sortedpos = jax.lax.dynamic_slice(
                    sorted_point_positions, (start_idx,), (bin_capacity,)
                )
                cell_bincount = end_idx - start_idx
                cell_indices = jnp.where(
                    jnp.arange(bin_capacity) < cell_bincount,
                    sortedpos,
                    -1,
                )
                return cell_indices

            grid = impl(start_idx, csum)
            grid = jnp.concatenate((grid, last_elt), axis=0)
            return grid

        self.__grid = make_grid(bin_counts)

    @partial(jax.jit, static_argnums=(0,))
    def get_bin_nd(self, point):
        """Get n-dimensional bin coordinates for a point"""
        return jnp.floor((jnp.array(point) - self.__lower) / self.__binsize).astype(int)

    def make_query_fn(self) -> None:
        binsize = np.asarray(self.__binsize)
        grid_shape = np.asarray(self.__grid_shape)
        grid = jnp.asarray(self.__grid)
        data = jnp.asarray(self.__data)

        def query_impl(xquery, k, distance_upper_bound):
            """
            Query the spatial grid for k nearest neighbors within radius.

            Args:
                xquery: Query point
                k: Number of neighbors to return
                qradius: Search radius

            Returns:
                Tuple of (distances, indices) to nearest neighbors
            """
            qrange = int(np.ceil(distance_upper_bound / binsize))
            qlower = self.get_bin_nd(xquery) - qrange

            query_bins = jnp.meshgrid(*[jnp.arange(qrange * 2) + ql for ql in qlower])
            query_bins = jnp.stack([q.flatten() for q in query_bins], axis=-1)

            in_bounds = jnp.all((query_bins >= 0) & (query_bins < jnp.array(grid_shape)), axis=1)
            query_ids = jnp.ravel_multi_index(query_bins.T, grid_shape, mode="clip")  # type: ignore
            query_ids = jnp.where(in_bounds, query_ids, -1)
            candidates = grid[query_ids].flatten()
            candidates = jnp.pad(candidates, (0, k), constant_values=-1)
            candidate_positions = data[candidates]

            sqdists = jnp.sum(jnp.square(candidate_positions - xquery), axis=1)

            mask = (sqdists < distance_upper_bound**2) & (candidates != -1)
            sqdists = jnp.where(mask, sqdists, jnp.inf)
            candidates = jnp.where(mask, candidates, -1)
            topk_dist, topk_ids = jax.lax.top_k(-sqdists, k=k)
            topk = candidates[topk_ids]
            topk_dist = jnp.sqrt(-topk_dist)
            topk_dist = jnp.where(topk != -1, topk_dist, jnp.inf)

            return topk_dist, topk

        jitquery = jit(vmap(query_impl, in_axes=(0, None, None)), static_argnums=(1, 2))

        def query(xquery, k, distance_upper_bound):
            return jitquery(xquery, k, distance_upper_bound)

        self.query = query


def get_knn_std(x, y, tree, **kw):
    """
    Get the k-nearest neighbors of x in the tree,
    and return their weighted standard deviation.
    """

    indices, weights = get_gaussian_weighted_knn(x, tree, **kw)
    assert indices.shape == weights.shape
    normed_w = weights / weights.sum(axis=1)[:, None]
    weighted_mean = (y[indices] * normed_w[:, :, None]).sum(axis=1)

    # Compute weighted variance (and then std)
    squared_diff = (y[indices] - weighted_mean[:, None, :]) ** 2
    weighted_squared_diff = squared_diff * normed_w[:, :, None]
    variance = weighted_squared_diff.sum(axis=1)

    return jnp.sqrt(variance)


def get_knn_quantile(x, y, tree, qu, **kw):
    import jax

    indices, weights = get_gaussian_weighted_knn(x, tree, **kw)
    q = jax.vmap(weighted_quantile, in_axes=(0, 0, None))(y[indices], weights, qu)
    density = np.nansum(weights, axis=1)
    return q, density


@jax.jit
def weighted_quantile(data, weights, qu):
    ix = jnp.argsort(data)
    data = data[ix]
    weights = weights[ix]
    cdf = (jnp.cumsum(weights) - 0.5 * weights) / jnp.sum(weights)
    return jnp.interp(qu, cdf, data)


def jax_gausspdf(x, mu, sigma):
    return jax.scipy.stats.norm.pdf(x, loc=mu, scale=sigma)
