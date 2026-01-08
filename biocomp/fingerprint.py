"""Deterministic network fingerprinting for design verification.

Generates unique signatures for networks by running deterministic forward passes
on canonical input grids. Used to detect discrepancies between logged designs
during training and final committed/saved models.

Example:
    >>> from biocomp.fingerprint import compute_fingerprint
    >>> nm = NetworkModel(model=model, network=committed_network)
    >>> fingerprint = compute_fingerprint(nm)
    >>> print(f"Network fingerprint: {fingerprint}")
"""

import hashlib
import numpy as np
import jax
from typing import TYPE_CHECKING

from biocomp.logging_config import get_logger

if TYPE_CHECKING:
    from biocomp.compute import ComputeStack
    from biocomp.parameters import ParameterTree

logger = get_logger(__name__)

FINGERPRINT_SEED = 42
FINGERPRINT_RESOLUTION = 21
FINGERPRINT_DECIMALS = 4


def _generate_canonical_grid(
    n_inputs: int, resolution: int, seed: int = FINGERPRINT_SEED
) -> np.ndarray:
    """Generate canonical grid in [0,1]^n_inputs for deterministic evaluation.

    For 1-3 inputs, creates a regular meshgrid. For higher dimensions,
    falls back to deterministic quasi-random sampling.

    Args:
        n_inputs: Number of input dimensions
        resolution: Grid points per dimension
        seed: Random seed for high-dimensional fallback

    Returns:
        Array of shape (n_points, n_inputs) with values in [0, 1]
    """
    if n_inputs == 1:
        return np.linspace(0, 1, resolution).reshape(-1, 1).astype(np.float32)
    elif n_inputs == 2:
        vals = np.linspace(0, 1, resolution)
        g1, g2 = np.meshgrid(vals, vals)
        return np.stack([g1.ravel(), g2.ravel()], axis=1).astype(np.float32)
    elif n_inputs == 3:
        vals = np.linspace(0, 1, resolution)
        g1, g2, g3 = np.meshgrid(vals, vals, vals)
        return np.stack([g1.ravel(), g2.ravel(), g3.ravel()], axis=1).astype(np.float32)
    else:
        n_points = resolution ** min(n_inputs, 3)
        return np.asarray(
            jax.random.uniform(jax.random.PRNGKey(seed), (n_points, n_inputs))
        ).astype(np.float32)


def _hash_output(Y: np.ndarray, decimals: int = FINGERPRINT_DECIMALS) -> str:
    """Hash rounded output array to 16-char hex digest.

    Rounds to specified decimal places before hashing to allow for
    small numerical differences (approximate matching).

    Args:
        Y: Output array to hash
        decimals: Decimal places to round to before hashing

    Returns:
        16-character hex digest (truncated SHA256)
    """
    Y_rounded = np.round(np.asarray(Y, dtype=np.float32), decimals=decimals)
    return hashlib.sha256(Y_rounded.tobytes()).hexdigest()[:16]


def compute_fingerprint(
    network_model,
    network_idx: int = 0,
    resolution: int = FINGERPRINT_RESOLUTION,
    seed: int = FINGERPRINT_SEED,
    decimals: int = FINGERPRINT_DECIMALS,
) -> str:
    """Compute fingerprint for a committed network via NetworkModel.

    Runs a deterministic forward pass on a canonical input grid and hashes
    the output. The fingerprint is independent of compute stack layer ordering
    because it only considers the specific network's output.

    Args:
        network_model: NetworkModel wrapping the committed network(s)
        network_idx: Which network in the stack to fingerprint (default: 0)
        resolution: Grid points per input dimension (21 → 441 for 2D)
        seed: Fixed seed for determinism
        decimals: Decimal places for approximate matching

    Returns:
        16-character hex digest fingerprint
    """
    from biocomptools.modelmodel import NetworkModel

    networks = network_model.stack.networks
    network = networks[network_idx]
    n_inputs = network.nb_inputs
    X = _generate_canonical_grid(n_inputs, resolution, seed)

    if len(networks) == 1:
        nm = network_model
    else:
        nm = NetworkModel(model=network_model.model, network=network)

    Y, _ = nm.predict(
        X,
        key=jax.random.PRNGKey(seed),
        disable_variational=True,
        z_value=0.0,
    )

    return _hash_output(Y, decimals)


def compute_fingerprint_from_params(
    stack: "ComputeStack",
    params: "ParameterTree",
    model,
    rep_id: int = 0,
    target_id: int = 0,
    network_idx: int = 0,
    resolution: int = FINGERPRINT_RESOLUTION,
    seed: int = FINGERPRINT_SEED,
    decimals: int = FINGERPRINT_DECIMALS,
) -> str:
    """Compute fingerprint from design params (during training).

    Extracts params for specific (rep_id, target_id), commits to network,
    then computes fingerprint via NetworkModel prediction. This allows
    fingerprinting designs during optimization before they're saved.

    Args:
        stack: ComputeStack from design manager
        params: Full parameter tree with shape (n_replicates, n_targets, ...)
        model: BiocompModel with shared params
        rep_id: Replicate index to fingerprint
        target_id: Target index to fingerprint
        network_idx: Network index within scaffold
        resolution: Grid resolution
        seed: Fixed seed
        decimals: Decimal places for rounding

    Returns:
        16-character hex digest fingerprint
    """
    from biocomptools.modelmodel import NetworkModel

    specific_params = jax.tree.map(lambda x, r=rep_id, t=target_id: x[r, t], params)
    committed_networks = stack.commit(specific_params)
    network = committed_networks[network_idx]
    network_model = NetworkModel(model=model, network=network)

    return compute_fingerprint(
        network_model,
        network_idx=0,
        resolution=resolution,
        seed=seed,
        decimals=decimals,
    )


def compare_fingerprints(
    fp1: str,
    fp2: str,
    context: str = "",
) -> bool:
    """Compare two fingerprints and log warning if they differ.

    Args:
        fp1: First fingerprint
        fp2: Second fingerprint
        context: Optional context string for log messages

    Returns:
        True if fingerprints match, False otherwise
    """
    if fp1 == fp2:
        logger.debug(f"Fingerprints match{': ' + context if context else ''}: {fp1}")
        return True
    else:
        logger.warning(
            f"FINGERPRINT MISMATCH{': ' + context if context else ''}: fp1={fp1}, fp2={fp2}"
        )
        return False
