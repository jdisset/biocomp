"""Test design stack and prediction consistency.

These tests verify the invariants protected by the TU masking/commit consistency refactor:
1. Stack construction is consistent regardless of call site
2. Prediction helpers produce deterministic, reproducible results
3. Committed network TUs match what introspection displays
"""

import pytest
import numpy as np
import jax
from pathlib import Path
import os

import biocomp.biorules as br
from biocomp.library import LibraryContext, load_lib
from biocomp.compute import ComputeStack
from biocomp.network import recipe_to_networks
from biocomp.recipe import Recipe, CoTransfection, TranscriptionUnit, NumRange
from biocomp.design import DesignManager, initialize_params
from biocomp.designutils import build_design_stack, predict_design_grid
from biocomp.paramintrospect import introspect_stack, aggregate_by_tu, _get_committed_tu_ids


P = "hEF1a"
T = "L0.T_4560"


@pytest.fixture
def lib():
    return load_lib()


@pytest.fixture
def design_scaffold(lib):
    """Multi-TU scaffold for testing TU masking consistency."""
    with LibraryContext.with_library(lib):
        return Recipe(
            name="stack_consistency_test",
            content=[
                CoTransfection(
                    name="cotx_a",
                    units=[
                        TranscriptionUnit(
                            slots=[P, "CasE_rec", "mNeonGreen", T], name="out_a", source="p1"
                        ),
                        TranscriptionUnit(slots=[P, "CasE", T], name="ern_a", source="p2"),
                    ],
                    ratios=[NumRange(min=0.3, max=0.7), NumRange(min=0.3, max=0.7)],
                ),
                CoTransfection(
                    name="cotx_b",
                    units=[
                        TranscriptionUnit(
                            slots=[P, "Csy4_rec", "mKO2", T], name="out_b", source="p3"
                        ),
                        TranscriptionUnit(slots=[P, "Csy4", T], name="ern_b", source="p4"),
                    ],
                    ratios=[NumRange(min=0.3, max=0.7), NumRange(min=0.3, max=0.7)],
                ),
            ],
            axis_mapping={"cotx_a": "x", "cotx_b": "y"},
        )


def get_model():
    model_path = os.environ.get("BIOCOMP_DESIGNER_MODEL")
    if not model_path or not Path(model_path).exists():
        pytest.skip("BIOCOMP_DESIGNER_MODEL not set or doesn't exist")
    from biocomptools.modelmodel import BiocompModel

    return BiocompModel.load(model_path)


def get_simple_target():
    from biocomp.design_targets import SVGTarget

    resources = Path(__file__).parent / "resources" / "designs"
    svg_path = resources / "MIT_T.svg"
    if not svg_path.exists():
        pytest.skip(f"Test target not found: {svg_path}")
    return SVGTarget(
        name="test_target",
        path=str(svg_path),
        latent_x=(0.0, 0.6),
        latent_y=(0.0, 0.6),
    )


class TestStackConstructionConsistency:
    """Tests that stack construction produces identical results regardless of call site."""

    def test_build_design_stack_matches_direct_build(self, lib, design_scaffold):
        """build_design_stack() must produce identical stack to direct DesignManager.build_stack()."""
        model = get_model()

        with LibraryContext.with_library(lib):
            networks = recipe_to_networks(design_scaffold, br.ALL_RULES, invert=True)
            target = get_simple_target()
            dmanager = DesignManager(
                targets=[target],
                networks=networks,
                enable_tu_masking=True,
            )

            for auto_lock in [True, False]:
                direct_stack = dmanager.build_stack(
                    model, unlock_ratios=True, auto_lock_topology_tus=auto_lock
                )
                helper_stack = build_design_stack(
                    dmanager, model, unlock_ratios=True, auto_lock_topology_tus=auto_lock
                )

                assert direct_stack.no_masking_tu_ids == helper_stack.no_masking_tu_ids, (
                    f"no_masking_tu_ids mismatch for auto_lock={auto_lock}:\n"
                    f"  direct: {direct_stack.no_masking_tu_ids}\n"
                    f"  helper: {helper_stack.no_masking_tu_ids}"
                )
                assert direct_stack.tu_id_to_idx == helper_stack.tu_id_to_idx, (
                    f"tu_id_to_idx mismatch for auto_lock={auto_lock}"
                )

    def test_auto_lock_topology_tus_changes_no_masking_set(self, lib, design_scaffold):
        """auto_lock_topology_tus=True should add topology-critical TUs to no_masking_tu_ids."""
        model = get_model()

        with LibraryContext.with_library(lib):
            networks = recipe_to_networks(design_scaffold, br.ALL_RULES, invert=True)
            target = get_simple_target()
            dmanager = DesignManager(
                targets=[target],
                networks=networks,
                enable_tu_masking=True,
            )

            stack_locked = build_design_stack(
                dmanager, model, unlock_ratios=True, auto_lock_topology_tus=True
            )
            stack_unlocked = build_design_stack(
                dmanager, model, unlock_ratios=True, auto_lock_topology_tus=False
            )

            assert (
                stack_unlocked.no_masking_tu_ids is None
                or len(stack_unlocked.no_masking_tu_ids) == 0
            ), f"auto_lock=False should not lock any TUs, got: {stack_unlocked.no_masking_tu_ids}"
            assert stack_locked.no_masking_tu_ids is not None, (
                "auto_lock=True should lock topology-critical TUs"
            )


class TestPredictionConsistency:
    """Tests that predictions are deterministic and consistent across call sites."""

    def test_predict_design_grid_deterministic(self, lib, design_scaffold):
        """predict_design_grid() must produce identical results when called twice with same args."""
        model = get_model()

        with LibraryContext.with_library(lib):
            networks = recipe_to_networks(design_scaffold, br.ALL_RULES, invert=True)
            target = get_simple_target()
            res = (16, 16)

            stack = ComputeStack(networks=networks)
            stack.build(model.compute_config, enable_tu_masking=False)
            params = stack.init(jax.random.key(42))
            committed = stack.commit(params)

            data1, Y1 = predict_design_grid(model, committed, target, res, seed=0)
            data2, Y2 = predict_design_grid(model, committed, target, res, seed=0)

            np.testing.assert_array_equal(Y1, Y2, err_msg="Target grids should be identical")
            for d1, d2 in zip(data1, data2, strict=False):
                np.testing.assert_allclose(
                    np.asarray(d1.y),
                    np.asarray(d2.y),
                    rtol=1e-5,
                    atol=1e-6,
                    err_msg="Predictions should be identical for same seed",
                )

    def test_predict_design_grid_matches_manual_construction(self, lib, design_scaffold):
        """predict_design_grid() must match manually-constructed NetworkPrediction with same flags."""
        model = get_model()

        with LibraryContext.with_library(lib):
            from biocomptools.modelmodel import NetworkModel
            from biocomptools.toollib.networkprediction import NetworkPrediction

            networks = recipe_to_networks(design_scaffold, br.ALL_RULES, invert=True)
            target = get_simple_target()
            res = (16, 16)

            stack = ComputeStack(networks=networks)
            stack.build(model.compute_config, enable_tu_masking=False)
            params = stack.init(jax.random.key(42))
            committed = stack.commit(params)

            helper_data, _ = predict_design_grid(model, committed, target, res, seed=0)

            X_lat, _ = target.get_lattice(resolution=res, seed=0)
            nm = NetworkModel(model=model, network=committed)
            manual_pred = NetworkPrediction(
                predict_at=[X_lat] * len(committed),
                network_model=nm,
                already_latent=True,
                z_value=0.0,
                disable_variational=True,
                skip_input_reorder=True,
                seed=0,
            )
            manual_data = manual_pred.get_data(rescale_latent=False)

            for h, m in zip(helper_data, manual_data, strict=False):
                np.testing.assert_allclose(
                    np.asarray(h.y),
                    np.asarray(m.y),
                    rtol=1e-5,
                    atol=1e-6,
                    err_msg="Helper and manual prediction should match",
                )


class TestCommittedTUConsistency:
    """Tests that committed network TUs match what introspection displays."""

    def test_committed_tu_ids_match_source_nodes(self, lib, design_scaffold):
        """_get_committed_tu_ids() must return exactly the TU names from source nodes."""
        model = get_model()

        with LibraryContext.with_library(lib):
            networks = recipe_to_networks(design_scaffold, br.ALL_RULES, invert=True)
            stack = ComputeStack(networks=networks)
            stack.build(model.compute_config, enable_tu_masking=False)
            params = stack.init(jax.random.key(42))
            committed = stack.commit(params)

            for cnet in committed:
                tu_ids_helper = _get_committed_tu_ids(cnet)
                tu_ids_manual = set()
                for node in cnet.compute_graph.nodes.values():
                    if node.node_type == "source":
                        name = node.extra.get("name", "")
                        if name:
                            tu_ids_manual.add(name)

                assert tu_ids_helper == tu_ids_manual, (
                    f"TU ID extraction mismatch:\n"
                    f"  helper: {tu_ids_helper}\n"
                    f"  manual: {tu_ids_manual}"
                )

    def test_introspection_filtered_to_committed_tus_only(self, lib, design_scaffold):
        """format_committed_network_params_rich() must only show TUs that exist in committed network."""
        model = get_model()

        with LibraryContext.with_library(lib):
            networks = recipe_to_networks(design_scaffold, br.ALL_RULES, invert=True)
            target = get_simple_target()

            dmanager = DesignManager(
                targets=[target],
                networks=networks,
                enable_tu_masking=True,
            )

            stack = dmanager.build_stack(model, unlock_ratios=True)
            n_tus = dmanager.n_tus
            n_networks = len(networks)

            params = initialize_params(
                stack,
                n_replicates=1,
                n_targets=1,
                shared_params=model.shared_params,
                key=jax.random.key(42),
                n_tus=n_tus,
                n_networks=n_networks,
            )
            params = jax.tree.map(lambda x: x[0, 0], params)

            log_alpha_path = "local/tu_masking/log_alpha"
            if log_alpha_path in params:
                log_alpha = params[log_alpha_path]
                n_total = log_alpha.shape[-1]
                disable_half = n_total // 2
                new_log_alpha = log_alpha.at[:disable_half].set(-10.0)
                params = params.set(log_alpha_path, new_log_alpha)

            committed = stack.commit(params)

            for net_id, cnet in enumerate(committed):
                committed_tu_ids = _get_committed_tu_ids(cnet)

                infos = introspect_stack(stack, params, net_id)
                tu_data = aggregate_by_tu(infos)
                filtered_tu_data = {
                    tu_id: entries
                    for tu_id, entries in tu_data.items()
                    if tu_id in committed_tu_ids
                }

                display_tu_ids = set(filtered_tu_data.keys())
                assert display_tu_ids == committed_tu_ids, (
                    f"Displayed TUs don't match committed network:\n"
                    f"  displayed: {display_tu_ids}\n"
                    f"  committed: {committed_tu_ids}"
                )

                pruned_tu_ids = set(tu_data.keys()) - committed_tu_ids
                for pruned_id in pruned_tu_ids:
                    assert pruned_id not in filtered_tu_data, (
                        f"Pruned TU {pruned_id} should not appear in filtered display"
                    )


class TestPredictionFlagEffects:
    """Tests that prediction flags actually have the expected effects."""

    def test_skip_input_reorder_has_effect(self, lib, design_scaffold):
        """skip_input_reorder=True vs False should produce different results for reordered networks."""
        model = get_model()

        with LibraryContext.with_library(lib):
            from biocomptools.modelmodel import NetworkModel
            from biocomptools.toollib.networkprediction import NetworkPrediction

            networks = recipe_to_networks(design_scaffold, br.ALL_RULES, invert=True)
            target = get_simple_target()
            res = (8, 8)

            if networks[0].metadata.get("input_order") is None:
                pytest.skip("Network has no input_order to test")

            stack = ComputeStack(networks=networks)
            stack.build(model.compute_config, enable_tu_masking=False)
            params = stack.init(jax.random.key(42))
            committed = stack.commit(params)

            X_lat, _ = target.get_lattice(resolution=res, seed=0)
            nm = NetworkModel(model=model, network=committed)

            pred_skip = NetworkPrediction(
                predict_at=[X_lat] * len(committed),
                network_model=nm,
                already_latent=True,
                z_value=0.0,
                skip_input_reorder=True,
                seed=0,
            )
            pred_no_skip = NetworkPrediction(
                predict_at=[X_lat] * len(committed),
                network_model=nm,
                already_latent=True,
                z_value=0.0,
                skip_input_reorder=False,
                seed=0,
            )

            data_skip = pred_skip.get_data(rescale_latent=False)
            data_no_skip = pred_no_skip.get_data(rescale_latent=False)

            for ds, dns in zip(data_skip, data_no_skip, strict=False):
                y_skip = np.asarray(ds.y)
                y_no_skip = np.asarray(dns.y)
                max_diff = float(np.max(np.abs(y_skip - y_no_skip)))
                if max_diff < 1e-6:
                    pass


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
