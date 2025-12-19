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


SamplingConfigUnion = Union[UniformSampling, LatticeSampling]


class TargetBase(BaseModel, ABC):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: Optional[str] = None
    lattice_x_extent: tuple[float, float] = (0.0, 1.0)
    lattice_y_extent: tuple[float, float] = (0.0, 1.0)

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
    img_latent_xlim: tuple[float, float] = (0.0, 1.0)
    img_latent_ylim: tuple[float, float] = (0.0, 1.0)
    img_latent_outlim: tuple[float, float] = (0.0, 1.0)
    transform_to_log_space: bool = False
    max_is_black: bool = True

    @model_validator(mode="after")
    def _auto_log_extent(self):
        if self.transform_to_log_space:
            if self.lattice_x_extent == (0.0, 1.0):
                self.lattice_x_extent = (0.1, 1.0)
            if self.lattice_y_extent == (0.0, 1.0):
                self.lattice_y_extent = (0.1, 1.0)
        return self

    def _sample(self, n: int, seed: int, grid: Optional[tuple[int, int]]):
        return sample_from_svg(
            self.path,
            n=n,
            seed=seed,
            grid=grid,
            grid_jitter_std=0.0,
            log=self.transform_to_log_space,
            lattice_x_extent=self.lattice_x_extent,
            lattice_y_extent=self.lattice_y_extent,
            img_latent_xlim=self.img_latent_xlim,
            img_latent_ylim=self.img_latent_ylim,
            img_latent_outlim=self.img_latent_outlim,
            max_is_black=self.max_is_black,
        )

    def get_lattice(
        self, resolution: tuple[int, int], seed: int = 0
    ) -> tuple[np.ndarray, np.ndarray]:
        X, Y = self._sample(n=1, seed=seed, grid=resolution)
        return X, Y[0]

    def sample_uniform(self, n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
        return self._sample(n=n, seed=seed, grid=None)


class Target(SVGTarget):
    """Legacy alias for SVGTarget."""

    xlim: Optional[tuple[float, float]] = None
    ylim: Optional[tuple[float, float]] = None
    outlim: Optional[tuple[float, float]] = None
    rescale_to: Optional[dict] = None

    @model_validator(mode="after")
    def _migrate_legacy_params(self):
        def _warn(old, new):
            warnings.warn(
                f"Target.{old} is deprecated, use {new} instead", DeprecationWarning, stacklevel=3
            )

        if self.xlim is not None:
            _warn("xlim", "lattice_x_extent")
            self.lattice_x_extent = self.xlim
        if self.ylim is not None:
            _warn("ylim", "lattice_y_extent")
            self.lattice_y_extent = self.ylim
        if self.outlim is not None:
            _warn("outlim", "img_latent_outlim")
            self.img_latent_outlim = self.outlim
        if self.rescale_to is not None:
            _warn("rescale_to", "lattice_*_extent and img_latent_*lim")
            if "x" in self.rescale_to:
                self.lattice_x_extent = tuple(self.rescale_to["x"])
            if "y" in self.rescale_to:
                self.lattice_y_extent = tuple(self.rescale_to["y"])
            if "out" in self.rescale_to:
                self.img_latent_outlim = tuple(self.rescale_to["out"])
        return self


class DataTarget(TargetBase):
    X: np.ndarray
    Y: np.ndarray
    z_slice: Optional[float] = None
    z_tolerance: float = 0.05
    original_network: Optional[Network] = None
    _lattice_X: Optional[np.ndarray] = None
    _lattice_Y: Optional[np.ndarray] = None

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
            xlims=self.lattice_x_extent,
            ylims=self.lattice_y_extent,
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
