"""Comprehensive tests for JAX-native schedule system.

Tests focus on:
1. Correctness of schedule evaluation vs optax equivalents
2. Recompilation behavior (critical for hyperopt)
3. Edge cases and boundary conditions
4. Integration with loss function infrastructure
"""

import time
import pytest
import jax
import jax.numpy as jnp
import numpy as np
import optax

from biocomp.optimutils import (
    jax_three_phase_schedule,
    jax_linear_schedule,
    three_phase_schedule,
)
from biocomp.designloss import (
    normalize_schedule_spec,
    init_schedule_params,
    _get_schedule_value,
    HYPEROPT_SCHEDULE_NAMESPACE,
)
from biocomp.parameters import ParameterTree


class TestJaxThreePhaseScheduleCorrectness:
    """Test that JAX schedule produces same values as optax schedule."""

    @pytest.mark.parametrize(
        "phase1_frac,phase2_frac",
        [
            (0.2, 0.6),
            (0.4, 0.75),
            (0.1, 0.9),
            (0.3, 0.5),
        ],
    )
    @pytest.mark.parametrize(
        "phase1_value,phase2_end,phase3_end",
        [
            (1.0, 0.5, 0.1),
            (0.0, 0.5, 1.0),  # increasing schedule
            (0.5, 0.5, 0.5),  # constant throughout
            (1.0, 0.01, 0.001),
        ],
    )
    def test_matches_optax_schedule(
        self, phase1_frac, phase2_frac, phase1_value, phase2_end, phase3_end
    ):
        """JAX schedule must match optax three_phase_schedule at all test points."""
        total_steps = 1000

        optax_sched = three_phase_schedule(
            total_steps,
            phase1_frac,
            phase2_frac,
            phase1_value,
            phase2_end,
            phase3_end,
        )

        test_steps = [0, 1, 10, 100, 250, 500, 750, 900, 999]
        for step in test_steps:
            optax_val = float(optax_sched(step))
            jax_val = float(
                jax_three_phase_schedule(
                    step,
                    total_steps,
                    phase1_frac,
                    phase2_frac,
                    phase1_value,
                    phase2_end,
                    phase3_end,
                )
            )
            np.testing.assert_allclose(
                jax_val,
                optax_val,
                rtol=1e-5,
                err_msg=f"Mismatch at step {step}: jax={jax_val}, optax={optax_val}",
            )

    def test_phase_boundaries_correct(self):
        """Verify schedule values at exact phase boundaries."""
        total_steps = 100
        phase1_frac, phase2_frac = 0.3, 0.7
        p1_val, p2_end, p3_end = 2.0, 1.0, 0.5

        phase1_end_step = int(phase1_frac * total_steps)  # 30
        phase2_end_step = int(phase2_frac * total_steps)  # 70

        # at phase1_end (step 30), should still be in phase1 (value = p1_val)
        val_at_p1_end = float(
            jax_three_phase_schedule(
                phase1_end_step - 1, total_steps, phase1_frac, phase2_frac, p1_val, p2_end, p3_end
            )
        )
        assert val_at_p1_end == pytest.approx(p1_val, rel=1e-4)

        # just after phase1_end, should start decaying
        val_after_p1 = float(
            jax_three_phase_schedule(
                phase1_end_step + 1, total_steps, phase1_frac, phase2_frac, p1_val, p2_end, p3_end
            )
        )
        assert val_after_p1 < p1_val  # should be decaying

        # at phase2_end (step 70), should be close to p2_end
        val_at_p2_end = float(
            jax_three_phase_schedule(
                phase2_end_step, total_steps, phase1_frac, phase2_frac, p1_val, p2_end, p3_end
            )
        )
        assert val_at_p2_end == pytest.approx(p2_end, rel=0.1)


class TestJaxLinearScheduleCorrectness:
    """Test linear schedule correctness."""

    def test_linear_interpolation(self):
        """Linear schedule should interpolate correctly."""
        total_steps = 100
        start, end = 1.0, 0.0

        assert jax_linear_schedule(0, total_steps, start, end) == pytest.approx(start)
        assert jax_linear_schedule(50, total_steps, start, end) == pytest.approx(0.5)
        assert jax_linear_schedule(100, total_steps, start, end) == pytest.approx(end)

    def test_matches_optax_polynomial(self):
        """Should match optax.polynomial_schedule with power=1."""
        total_steps = 1000
        start, end = 2.0, 0.5

        optax_sched = optax.polynomial_schedule(
            init_value=start, end_value=end, power=1.0, transition_steps=total_steps
        )

        for step in [0, 100, 500, 999]:
            jax_val = float(jax_linear_schedule(step, total_steps, start, end))
            optax_val = float(optax_sched(step))
            np.testing.assert_allclose(jax_val, optax_val, rtol=1e-5)


class TestNoRecompilation:
    """Critical tests: verify no recompilation when changing schedule params."""

    def test_jax_schedule_no_recompile(self):
        """Changing JAX schedule params must NOT trigger recompilation."""
        total_steps = 100

        @jax.jit
        def eval_schedule(step, p1_frac, p2_frac, p1_val, p2_end, p3_end):
            return jax_three_phase_schedule(
                step, total_steps, p1_frac, p2_frac, p1_val, p2_end, p3_end
            )

        # first call - compiles
        t0 = time.perf_counter()
        _ = eval_schedule(50.0, 0.4, 0.75, 1.0, 0.5, 0.1)
        t_compile = time.perf_counter() - t0

        # second call with different params
        t0 = time.perf_counter()
        _ = eval_schedule(50.0, 0.3, 0.6, 2.0, 0.8, 0.2)
        t_reuse = time.perf_counter() - t0

        # reuse should be at least 100x faster than initial compile
        assert t_reuse < t_compile / 50, (
            f"Possible recompilation: compile={t_compile:.4f}s, reuse={t_reuse:.4f}s"
        )

    def test_optax_schedule_does_recompile(self):
        """Verify that optax schedules DO recompile (baseline for comparison)."""
        total_steps = 100

        def make_jitted_optax(p1_frac, p2_frac, p1_val, p2_end, p3_end):
            sched = three_phase_schedule(total_steps, p1_frac, p2_frac, p1_val, p2_end, p3_end)

            @jax.jit
            def eval_sched(step):
                return sched(step)

            return eval_sched

        fn1 = make_jitted_optax(0.4, 0.75, 1.0, 0.5, 0.1)
        t0 = time.perf_counter()
        _ = fn1(50)
        t_compile1 = time.perf_counter() - t0

        fn2 = make_jitted_optax(0.3, 0.6, 2.0, 0.8, 0.2)
        t0 = time.perf_counter()
        _ = fn2(50)
        t_compile2 = time.perf_counter() - t0

        # both should be similar (both compile)
        ratio = t_compile2 / t_compile1
        assert 0.1 < ratio < 10, f"Expected similar times, got ratio={ratio:.2f}"


class TestNormalizeScheduleSpec:
    """Test schedule spec normalization."""

    def test_constant_spec(self):
        """Float/int becomes constant schedule (all phases same value)."""
        result = normalize_schedule_spec(0.5)
        assert result["phase1_value"] == 0.5
        assert result["phase2_end_value"] == 0.5
        assert result["phase3_end_value"] == 0.5
        assert result["phase1_frac"] == 0.0
        assert result["phase2_frac"] == 0.0

    def test_linear_spec(self):
        """Dict with start/end becomes linear schedule."""
        result = normalize_schedule_spec({"start": 1.0, "end": 0.1})
        assert result["phase1_value"] == 1.0
        assert result["phase2_end_value"] == 0.1
        assert result["phase3_end_value"] == 0.1
        assert result["phase1_frac"] == 0.0
        assert result["phase2_frac"] == 1.0

    def test_three_phase_spec(self):
        """Dict with phase1_value etc. uses explicit values."""
        result = normalize_schedule_spec(
            {
                "phase1_frac": 0.2,
                "phase2_frac": 0.6,
                "phase1_value": 2.0,
                "phase2_end_value": 1.0,
                "phase3_end_value": 0.5,
            }
        )
        assert result["phase1_frac"] == 0.2
        assert result["phase2_frac"] == 0.6
        assert result["phase1_value"] == 2.0
        assert result["phase2_end_value"] == 1.0
        assert result["phase3_end_value"] == 0.5

    def test_callable_passes_through(self):
        """Callable schedules pass through unchanged."""

        def my_schedule(step):
            return step * 0.1

        result = normalize_schedule_spec(my_schedule)
        assert result is my_schedule

    def test_invalid_spec_raises(self):
        """Invalid specs should raise clear errors."""
        with pytest.raises(ValueError, match="Invalid schedule spec"):
            normalize_schedule_spec({"foo": "bar"})
        with pytest.raises(ValueError):
            normalize_schedule_spec([1, 2, 3])


class TestInitScheduleParams:
    """Test schedule params initialization."""

    def test_creates_all_params(self):
        """init_schedule_params creates all expected param entries."""
        specs = {
            "lambda_l0": 0.01,  # constant
            "tu_temperature": {"start": 1.0, "end": 0.1},  # linear
        }
        params = init_schedule_params(specs)

        # constant lambda_l0
        assert f"{HYPEROPT_SCHEDULE_NAMESPACE}/lambda_l0_phase1_value" in params
        assert f"{HYPEROPT_SCHEDULE_NAMESPACE}/lambda_l0_phase1_frac" in params
        assert params[f"{HYPEROPT_SCHEDULE_NAMESPACE}/lambda_l0_phase1_value"] == 0.01

        # linear tu_temperature
        assert f"{HYPEROPT_SCHEDULE_NAMESPACE}/tu_temperature_phase1_value" in params
        assert params[f"{HYPEROPT_SCHEDULE_NAMESPACE}/tu_temperature_phase1_value"] == 1.0
        assert params[f"{HYPEROPT_SCHEDULE_NAMESPACE}/tu_temperature_phase2_end_value"] == 0.1

    def test_skips_callables(self):
        """Callable schedules are skipped (not stored in params)."""
        specs = {
            "lambda_l0": lambda step: 0.01,  # callable - skip
            "tu_temperature": 0.5,  # constant - include
        }
        params = init_schedule_params(specs)

        assert f"{HYPEROPT_SCHEDULE_NAMESPACE}/lambda_l0_phase1_value" not in params
        assert f"{HYPEROPT_SCHEDULE_NAMESPACE}/tu_temperature_phase1_value" in params


def _create_params_with_schedule(schedule_params: dict) -> ParameterTree:
    """Helper to create a ParameterTree with schedule params."""
    params = ParameterTree()
    for path, value in schedule_params.items():
        params[path] = value
    return params


class TestGetScheduleValue:
    """Test _get_schedule_value with both modes."""

    def test_optax_mode_with_none_ns(self):
        """With schedule_ns=None, uses as_schedule (optax mode)."""
        params = ParameterTree()
        val = _get_schedule_value(params, 50, 100, "test", 0.5, schedule_ns=None)
        assert float(val) == 0.5

    def test_dynamic_mode_with_params(self):
        """With schedule_ns set, reads from params tree."""
        schedule_params = init_schedule_params(
            {"test": {"phase1_value": 1.0, "phase3_end_value": 0.1}}
        )
        params = _create_params_with_schedule(schedule_params)

        # at step 0, should be close to phase1_value
        val_start = _get_schedule_value(
            params, 0, 100, "test", 999.0, schedule_ns=HYPEROPT_SCHEDULE_NAMESPACE
        )
        assert float(val_start) == pytest.approx(1.0, rel=0.01)

        # at final step, should be close to phase3_end_value
        val_end = _get_schedule_value(
            params, 100, 100, "test", 999.0, schedule_ns=HYPEROPT_SCHEDULE_NAMESPACE
        )
        assert float(val_end) == pytest.approx(0.1, rel=0.01)

    def test_fallback_when_param_missing(self):
        """Falls back to schedule_or_value when param not in tree."""
        params = ParameterTree()  # empty
        val = _get_schedule_value(
            params, 50, 100, "nonexistent", 0.123, schedule_ns=HYPEROPT_SCHEDULE_NAMESPACE
        )
        assert float(val) == pytest.approx(0.123)


class TestScheduleGradients:
    """Test that gradients flow through JAX schedules correctly."""

    def test_gradient_through_schedule(self):
        """Gradients should flow through jax_three_phase_schedule."""
        total_steps = 100

        def loss_fn(params):
            # params = [phase1_value, phase2_end_value, phase3_end_value]
            val = jax_three_phase_schedule(
                50.0, total_steps, 0.4, 0.75, params[0], params[1], params[2]
            )
            return val**2  # simple loss

        params = jnp.array([1.0, 0.5, 0.1])
        grads = jax.grad(loss_fn)(params)

        # at step 50 (past phase1), gradient should be non-zero for at least one param
        assert jnp.any(grads != 0), "Gradients should flow through schedule"

    def test_gradient_wrt_phase_fracs(self):
        """Gradients should flow through phase fraction params."""
        total_steps = 100

        def loss_fn(fracs):
            val = jax_three_phase_schedule(
                50.0,
                total_steps,
                fracs[0],
                fracs[1],  # phase1_frac, phase2_frac
                1.0,
                0.5,
                0.1,
            )
            return val

        fracs = jnp.array([0.4, 0.75])
        grads = jax.grad(loss_fn)(fracs)
        # step 50 is in phase2 (after phase1_frac*100=40), so gradient wrt phase1_frac should exist
        assert grads[0] != 0 or grads[1] != 0, "Gradients should flow through phase fracs"


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_total_steps(self):
        """Should handle edge case of very small total_steps gracefully."""
        # with total_steps=1, all phases collapse
        val = jax_three_phase_schedule(0, 1, 0.4, 0.75, 1.0, 0.5, 0.1)
        assert jnp.isfinite(val)

    def test_negative_step(self):
        """Negative step should clamp to phase1 value."""
        val = jax_three_phase_schedule(-10, 100, 0.4, 0.75, 1.0, 0.5, 0.1)
        assert float(val) == pytest.approx(1.0)  # phase1_value

    def test_step_beyond_total(self):
        """Step beyond total should clamp to phase3 end value."""
        val = jax_three_phase_schedule(200, 100, 0.4, 0.75, 1.0, 0.5, 0.1)
        assert float(val) == pytest.approx(0.1)  # phase3_end_value

    def test_phase_fracs_at_boundaries(self):
        """phase1_frac=0 and phase2_frac=1 should work (linear schedule)."""
        val_start = jax_three_phase_schedule(0, 100, 0.0, 1.0, 1.0, 0.0, 0.0)
        val_end = jax_three_phase_schedule(100, 100, 0.0, 1.0, 1.0, 0.0, 0.0)
        assert float(val_start) == pytest.approx(1.0)
        assert float(val_end) == pytest.approx(0.0)

    def test_inverted_schedule(self):
        """Schedule that increases over time should work correctly."""
        val_start = jax_three_phase_schedule(0, 100, 0.0, 1.0, 0.0, 1.0, 1.0)
        val_end = jax_three_phase_schedule(100, 100, 0.0, 1.0, 0.0, 1.0, 1.0)
        assert float(val_start) == pytest.approx(0.0)
        assert float(val_end) == pytest.approx(1.0)


class TestVmapCompatibility:
    """Test vmap over schedule params (crucial for hyperopt with vmap trials)."""

    def test_vmap_over_values(self):
        """Schedule should work under vmap over value params."""
        total_steps = 100

        def eval_schedule(p3_end):
            return jax_three_phase_schedule(80, total_steps, 0.4, 0.75, 1.0, 0.5, p3_end)

        p3_ends = jnp.array([0.1, 0.2, 0.3, 0.4])
        results = jax.vmap(eval_schedule)(p3_ends)

        # at step 80 (in phase3), results should reflect different p3_end values
        assert results.shape == (4,)
        assert jnp.all(jnp.diff(results) > 0)  # higher p3_end -> higher value

    def test_vmap_over_all_params(self):
        """Full vmap over all schedule params."""
        total_steps = 100

        @jax.vmap
        def eval_schedule_batch(params):
            # params: [step, p1_frac, p2_frac, p1_val, p2_end, p3_end]
            return jax_three_phase_schedule(
                params[0], total_steps, params[1], params[2], params[3], params[4], params[5]
            )

        batch = jnp.array(
            [
                [50.0, 0.4, 0.75, 1.0, 0.5, 0.1],
                [50.0, 0.3, 0.6, 2.0, 1.0, 0.5],
                [80.0, 0.4, 0.75, 1.0, 0.5, 0.1],
            ]
        )
        results = eval_schedule_batch(batch)
        assert results.shape == (3,)
        assert jnp.all(jnp.isfinite(results))


class TestHyperoptIntegration:
    """Integration tests verifying hyperparams affect loss computation."""

    def test_different_weights_produce_different_values(self):
        """Different schedule weights should produce different schedule values."""
        total_steps = 100
        step = 50

        params1 = _create_params_with_schedule(
            init_schedule_params(
                {
                    "w_sinkhorn": 0.1,
                    "w_lncc": 0.5,
                }
            )
        )
        params2 = _create_params_with_schedule(
            init_schedule_params(
                {
                    "w_sinkhorn": 2.0,
                    "w_lncc": 0.1,
                }
            )
        )

        w_sink1 = _get_schedule_value(
            params1, step, total_steps, "w_sinkhorn", 0.0, HYPEROPT_SCHEDULE_NAMESPACE
        )
        w_sink2 = _get_schedule_value(
            params2, step, total_steps, "w_sinkhorn", 0.0, HYPEROPT_SCHEDULE_NAMESPACE
        )

        assert float(w_sink1) == pytest.approx(0.1)
        assert float(w_sink2) == pytest.approx(2.0)
        assert abs(float(w_sink1) - float(w_sink2)) > 1.0

    def test_schedule_values_evolve_over_steps(self):
        """Three-phase schedule should change value over steps."""
        total_steps = 100

        params = _create_params_with_schedule(
            init_schedule_params(
                {
                    "lambda_l0": {
                        "phase1_frac": 0.3,
                        "phase2_frac": 0.7,
                        "phase1_value": 0.0,
                        "phase2_end_value": 0.05,
                        "phase3_end_value": 0.1,
                    }
                }
            )
        )

        val_early = _get_schedule_value(
            params, 10, total_steps, "lambda_l0", 0.0, HYPEROPT_SCHEDULE_NAMESPACE
        )
        val_mid = _get_schedule_value(
            params, 50, total_steps, "lambda_l0", 0.0, HYPEROPT_SCHEDULE_NAMESPACE
        )
        val_late = _get_schedule_value(
            params, 90, total_steps, "lambda_l0", 0.0, HYPEROPT_SCHEDULE_NAMESPACE
        )

        assert float(val_early) == pytest.approx(0.0, abs=0.01)
        assert float(val_mid) > float(val_early)
        assert float(val_late) > float(val_mid)

    def test_phase_frac_constraint_values(self):
        """Phase1_frac < phase2_frac ranges are enforced in YAML config."""

        yaml_path = "/home/jean/Code/biocompiler/biocomp-jobs/hyperopt/hyperparams/design_19.yaml"
        try:
            with open(yaml_path) as f:
                content = f.read()

            phase1_high = None
            phase2_low = None
            for line in content.split("\n"):
                if "name: phase1_frac" in line:
                    for next_line in content.split("\n")[content.split("\n").index(line) :]:
                        if "high:" in next_line:
                            phase1_high = float(next_line.split(":")[1].strip())
                            break
                if "name: phase2_frac" in line:
                    for next_line in content.split("\n")[content.split("\n").index(line) :]:
                        if "low:" in next_line:
                            phase2_low = float(next_line.split(":")[1].strip())
                            break

            if phase1_high is not None and phase2_low is not None:
                assert phase1_high < phase2_low, (
                    f"phase1_frac high ({phase1_high}) must be < phase2_frac low ({phase2_low})"
                )
        except FileNotFoundError:
            pytest.skip("Shared hyperparams file not found")

    def test_warning_on_missing_schedule(self, caplog):
        """Should log warning when schedule not found in params."""
        import logging
        from biocomp.designloss import _SCHEDULE_FALLBACK_WARNED

        _SCHEDULE_FALLBACK_WARNED.clear()

        params = ParameterTree()
        with caplog.at_level(logging.WARNING):
            val = _get_schedule_value(
                params, 50, 100, "missing_schedule", 0.5, HYPEROPT_SCHEDULE_NAMESPACE
            )

        assert float(val) == pytest.approx(0.5)
        assert any("missing_schedule" in record.message for record in caplog.records)
