# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jean Disset
import numpy as np
import pytest

pytest.importorskip("svgelements")
pytest.importorskip("svgpath2mpl")

from biocomp.design import DesignManager
from biocomp.design_targets import SVGTarget, LatticeSampling
from biocomp.graphengine import GraphState
from biocomp.network import Network


def _dummy_network(name="dummy"):
    return Network(name=name, compute_graph=GraphState(nodes={}, edges={}))


def _write_svg(tmp_path):
    svg = (
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"10\" height=\"10\">"
        "<rect width=\"10\" height=\"10\" fill=\"black\"/>"
        "</svg>"
    )
    path = tmp_path / "square.svg"
    path.write_text(svg)
    return path


def test_lattice_jitter_changes_coords(tmp_path):
    path = _write_svg(tmp_path)
    target = SVGTarget(path=str(path))

    sampling_no_jitter = LatticeSampling(resolution=(8, 8), jitter_std=0.0)
    sampling_jitter = LatticeSampling(resolution=(8, 8), jitter_std=0.1)

    dm_no = DesignManager(targets=[target], networks=[_dummy_network()], sampling=sampling_no_jitter)
    dm_yes = DesignManager(targets=[target], networks=[_dummy_network()], sampling=sampling_jitter)

    x_no, _ = dm_no.get_samples((1, 1), seed=123, share_across_networks=True)
    x_yes, _ = dm_yes.get_samples((1, 1), seed=123, share_across_networks=True)

    assert x_no[0].shape == x_yes[0].shape
    assert not np.allclose(np.asarray(x_no[0]), np.asarray(x_yes[0]))


def test_share_across_networks_stabilizes_noise(tmp_path):
    path = _write_svg(tmp_path)
    target = SVGTarget(path=str(path))
    sampling = LatticeSampling(resolution=(6, 6), noise_std=0.05)

    dm = DesignManager(
        targets=[target],
        networks=[_dummy_network("n0"), _dummy_network("n1")],
        sampling=sampling,
    )

    _, y_shared = dm.get_samples((2, 1), seed=999, share_across_networks=True)
    _, y_unshared = dm.get_samples((2, 1), seed=999, share_across_networks=False)

    assert np.allclose(np.asarray(y_shared[0]), np.asarray(y_shared[1]))
    assert not np.allclose(np.asarray(y_unshared[0]), np.asarray(y_unshared[1]))
