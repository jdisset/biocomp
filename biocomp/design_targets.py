"""Target classes and sampling configs for design optimization."""

import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal, Optional, Union

import numpy as np
from pydantic import BaseModel, ConfigDict, model_validator

from biocomp.utils import ArbitraryModel
from biocomp.network import Network
from biocomp.designutils import sample_from_svg, data_to_lattice_2d
from biocomp.logging_config import get_logger

logger = get_logger(__name__)

DEFAULT_RESCALE_TARGET = {"x": (0.0, 0.5), "y": (0.0, 0.5), "out": (0.09, 0.42)}


class SamplingConfig(ArbitraryModel):
    strategy: Literal["uniform", "lattice"] = "uniform"


class UniformSampling(SamplingConfig):
    strategy: Literal["uniform"] = "uniform"
    n_samples: int = 5000


class LatticeSampling(SamplingConfig):
    strategy: Literal["lattice"] = "lattice"
    resolution: tuple[int, int] = (64, 64)
    jitter_std: float = 0.0
    noise_std: float = 0.0


SamplingConfigUnion = Union[UniformSampling, LatticeSampling]


class TargetBase(BaseModel, ABC):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: Optional[str] = None
    latent_x: tuple[float, float] = (0.0, 0.6)
    latent_y: tuple[float, float] = (0.0, 0.6)

    @abstractmethod
    def get_lattice(
        self, resolution: tuple[int, int], seed: int = 0
    ) -> tuple[np.ndarray, np.ndarray]:
        """Returns (X_lattice, Y_lattice) where Y may contain NaN for out-of-data regions."""
        ...

    @abstractmethod
    def sample_uniform(self, n: int, seed: int) -> tuple[np.ndarray, np.ndarray]: ...


class SVGTarget(TargetBase):
    path: Union[str, Path]
    viewbox_x: tuple[float, float] = (0.0, 1.0)
    viewbox_y: tuple[float, float] = (0.0, 1.0)
    latent_out: tuple[float, float] = (0.0, 0.6)
    transform_to_log_space: bool = False
    max_is_black: bool = True
    blur_sigma: float = 3.0

    @model_validator(mode="after")
    def _auto_log_viewbox(self):
        if self.transform_to_log_space:
            if self.viewbox_x == (0.0, 1.0):
                self.viewbox_x = (0.1, 1.0)
            if self.viewbox_y == (0.0, 1.0):
                self.viewbox_y = (0.1, 1.0)
        return self

    def _sample(self, n: int, seed: int, grid: Optional[tuple[int, int]]):
        return sample_from_svg(
            self.path,
            n=n,
            seed=seed,
            grid=grid,
            grid_jitter_std=0.0,
            log=self.transform_to_log_space,
            viewbox_x=self.viewbox_x,
            viewbox_y=self.viewbox_y,
            latent_x=self.latent_x,
            latent_y=self.latent_y,
            latent_out=self.latent_out,
            max_is_black=self.max_is_black,
        )

    def get_lattice(
        self, resolution: tuple[int, int], seed: int = 0
    ) -> tuple[np.ndarray, np.ndarray]:
        X, Y = self._sample(n=1, seed=seed, grid=resolution)
        Y_out = Y[0]
        if self.blur_sigma > 0:
            from scipy.ndimage import gaussian_filter

            Y_out = gaussian_filter(Y_out, sigma=self.blur_sigma, mode="nearest")
        return X, Y_out

    def sample_uniform(self, n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
        return self._sample(n=n, seed=seed, grid=None)


class Target(SVGTarget):
    """Legacy alias for SVGTarget with deprecated parameter names."""

    xlim: Optional[tuple[float, float]] = None
    ylim: Optional[tuple[float, float]] = None
    outlim: Optional[tuple[float, float]] = None
    rescale_to: Optional[dict] = None
    lattice_x_extent: Optional[tuple[float, float]] = None
    lattice_y_extent: Optional[tuple[float, float]] = None
    img_latent_xlim: Optional[tuple[float, float]] = None
    img_latent_ylim: Optional[tuple[float, float]] = None
    img_latent_outlim: Optional[tuple[float, float]] = None

    @model_validator(mode="after")
    def _migrate_legacy_params(self):
        def _warn(old, new):
            warnings.warn(
                f"Target.{old} is deprecated, use {new} instead", DeprecationWarning, stacklevel=3
            )

        if self.lattice_x_extent is not None:
            _warn("lattice_x_extent", "viewbox_x")
            self.viewbox_x = self.lattice_x_extent
        if self.lattice_y_extent is not None:
            _warn("lattice_y_extent", "viewbox_y")
            self.viewbox_y = self.lattice_y_extent
        if self.img_latent_xlim is not None:
            _warn("img_latent_xlim", "latent_x")
            self.latent_x = self.img_latent_xlim
        if self.img_latent_ylim is not None:
            _warn("img_latent_ylim", "latent_y")
            self.latent_y = self.img_latent_ylim
        if self.img_latent_outlim is not None:
            _warn("img_latent_outlim", "latent_out")
            self.latent_out = self.img_latent_outlim
        if self.xlim is not None:
            _warn("xlim", "viewbox_x")
            self.viewbox_x = self.xlim
        if self.ylim is not None:
            _warn("ylim", "viewbox_y")
            self.viewbox_y = self.ylim
        if self.outlim is not None:
            _warn("outlim", "latent_out")
            self.latent_out = self.outlim
        if self.rescale_to is not None:
            _warn("rescale_to", "viewbox_*/latent_*")
            if "x" in self.rescale_to:
                self.viewbox_x = tuple(self.rescale_to["x"])
            if "y" in self.rescale_to:
                self.viewbox_y = tuple(self.rescale_to["y"])
            if "out" in self.rescale_to:
                self.latent_out = tuple(self.rescale_to["out"])
        return self


class DataTarget(TargetBase):
    X: np.ndarray
    Y: np.ndarray
    z_slice: Optional[float] = None
    z_tolerance: float = 0.05
    original_network: Optional[Network] = None
    scale_to_latent: bool = True  # If True, rescale X to fit in latent_x/latent_y
    _lattice_X: Optional[np.ndarray] = None
    _lattice_Y: Optional[np.ndarray] = None

    @model_validator(mode="after")
    def _rescale_x_to_latent(self):
        """Rescale X coordinates so the full data pattern fits in latent_x/latent_y.

        Without this, setting latent_x=(0, 0.6) when data spans (0, 1) would CROP
        the pattern. With rescaling, the full pattern is SCALED to fit.
        """
        if not self.scale_to_latent:
            return self

        X = np.asarray(self.X)
        if X.ndim != 2 or X.shape[1] < 2:
            return self

        # Compute data range for each dimension
        x_min, x_max = X[:, 0].min(), X[:, 0].max()
        y_min, y_max = X[:, 1].min(), X[:, 1].max()

        # Avoid division by zero for constant data
        x_range = x_max - x_min if x_max > x_min else 1.0
        y_range = y_max - y_min if y_max > y_min else 1.0

        # Scale X to fit in latent_x/latent_y
        X_scaled = X.copy()
        X_scaled[:, 0] = (X[:, 0] - x_min) / x_range * (
            self.latent_x[1] - self.latent_x[0]
        ) + self.latent_x[0]
        X_scaled[:, 1] = (X[:, 1] - y_min) / y_range * (
            self.latent_y[1] - self.latent_y[0]
        ) + self.latent_y[0]

        # Handle higher dimensions (just copy them)
        if X.shape[1] > 2:
            X_scaled[:, 2:] = X[:, 2:]

        self.X = X_scaled
        return self

    @classmethod
    def from_plot_data(cls, plot_data, rescaler=None, **kwargs):
        X, Y = np.asarray(plot_data.x), np.asarray(plot_data.y)
        if rescaler is not None:
            X, Y = rescaler.fwd(X), rescaler.fwd(Y)
        name = kwargs.pop("name", plot_data.metadata.get("network_name", "data_target"))
        return cls(X=X, Y=Y, name=name, **kwargs)

    def get_lattice(
        self, resolution: tuple[int, int], seed: int = 0, force_recompute: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        if self._lattice_X is not None and self._lattice_Y is not None and not force_recompute:
            return self._lattice_X, self._lattice_Y

        X_samples, Y_samples = data_to_lattice_2d(
            self.X,
            self.Y,
            xlims=self.latent_x,
            ylims=self.latent_y,
            resolution=resolution,
        )
        nan_mask = np.isnan(Y_samples)
        if nan_mask.any():
            valid_mean = np.nanmean(Y_samples)
            Y_samples = np.where(nan_mask, valid_mean, Y_samples)
            logger.debug(f"Filled {nan_mask.sum()} NaN values with mean={valid_mean:.4f}")

        self._lattice_X, self._lattice_Y = X_samples, Y_samples
        return self._lattice_X, self._lattice_Y

    def sample_uniform(self, n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(self.X), size=n, replace=True)
        Y_sampled = self.Y[indices]
        return self.X[indices], Y_sampled[:, None] if Y_sampled.ndim == 1 else Y_sampled


TargetUnion = Union[Target, SVGTarget, DataTarget]
