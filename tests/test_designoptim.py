import pytest
import jax
import jax.numpy as jnp

from biocomp.designoptim import (
    OptimPhase,
    OptimizationState,
    GradientDescentOptimizer,
    EvolutionaryOptimizer,
    HybridOptimizer,
)


def sphere(x: jnp.ndarray) -> float:
    return jnp.sum(x**2)


def rastrigin(x: jnp.ndarray) -> float:
    A = 10.0
    return A * x.shape[0] + jnp.sum(x**2 - A * jnp.cos(2 * jnp.pi * x))


def rosenbrock(x: jnp.ndarray) -> float:
    return jnp.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)


@pytest.fixture
def key():
    return jax.random.key(42)


class TestOptimizationState:
    def test_state_is_valid_pytree(self, key):
        state = OptimizationState(
            step=jnp.array(0, dtype=jnp.int32),
            params=jnp.ones(10),
            best_params=jnp.ones(10),
            best_loss=jnp.array(1.0),
            phase=jnp.array(OptimPhase.GD, dtype=jnp.int32),
            opt_state=None,
        )
        leaves, treedef = jax.tree_util.tree_flatten(state)
        assert isinstance(jax.tree_util.tree_unflatten(treedef, leaves), OptimizationState)

    def test_is_finite_true_for_valid_state(self, key):
        state = OptimizationState(
            step=jnp.array(0, dtype=jnp.int32),
            params=jnp.ones(10),
            best_params=jnp.ones(10),
            best_loss=jnp.array(1.0),
            phase=jnp.array(OptimPhase.GD, dtype=jnp.int32),
            opt_state=None,
        )
        assert state.is_finite()

    def test_is_finite_false_for_nan(self, key):
        state = OptimizationState(
            step=jnp.array(0, dtype=jnp.int32),
            params=jnp.ones(10),
            best_params=jnp.ones(10),
            best_loss=jnp.array(jnp.nan),
            phase=jnp.array(OptimPhase.GD, dtype=jnp.int32),
            opt_state=None,
        )
        assert not state.is_finite()


class TestGradientDescentOptimizer:
    def test_init_returns_valid_state(self, key):
        state = GradientDescentOptimizer(n_steps=100).init(key, jnp.ones(10), sphere)
        assert state.step == 0 and state.params.shape == (10,) and jnp.isfinite(state.best_loss)

    def test_reduces_loss_on_convex(self, key):
        opt = GradientDescentOptimizer(n_steps=200)
        state = opt.init(key, jnp.ones(10), sphere)
        initial_loss = float(state.best_loss)
        for _ in range(200):
            key, subkey = jax.random.split(key)
            state, _ = opt.step(state, subkey, sphere)
        assert float(state.best_loss) < initial_loss

    def test_tracks_best_correctly(self, key):
        opt = GradientDescentOptimizer(n_steps=50)
        state = opt.init(key, jnp.ones(10) * 5, sphere)
        for _ in range(50):
            key, subkey = jax.random.split(key)
            state, _ = opt.step(state, subkey, sphere)
        assert jnp.isclose(sphere(state.best_params), state.best_loss, rtol=0.01)

    def test_should_stop_at_n_steps(self, key):
        opt = GradientDescentOptimizer(n_steps=10)
        state = opt.init(key, jnp.ones(10), sphere)
        for _ in range(10):
            key, subkey = jax.random.split(key)
            state, _ = opt.step(state, subkey, sphere)
        assert opt.should_stop(state)

    def test_metrics_contain_loss_and_grad_norm(self, key):
        opt = GradientDescentOptimizer(n_steps=10)
        _, metrics = opt.step(opt.init(key, jnp.ones(10), sphere), key, sphere)
        assert "loss" in metrics and "grad_norm" in metrics
        assert jnp.isfinite(metrics["loss"]) and jnp.isfinite(metrics["grad_norm"])


class TestEvolutionaryOptimizer:
    def test_init_returns_valid_state(self, key):
        state = EvolutionaryOptimizer(population_size=16, n_generations=50).init(
            key, jnp.zeros(10), sphere
        )
        assert state.step == 0 and state.params.shape == (10,)
        assert jnp.isfinite(state.best_loss) and state.phase == OptimPhase.EC

    def test_reduces_loss(self, key):
        opt = EvolutionaryOptimizer(population_size=32, n_generations=50, sigma_init=0.5)
        state = opt.init(key, jnp.ones(10) * 2, sphere)
        initial_loss = float(state.best_loss)
        for _ in range(50):
            if opt.should_stop(state):
                break
            key, subkey = jax.random.split(key)
            state, _ = opt.step(state, subkey, sphere)
        assert float(state.best_loss) < initial_loss

    def test_auto_population_size(self, key):
        opt = EvolutionaryOptimizer(population_size="auto", n_generations=10)
        opt.init(key, jnp.zeros(100), sphere)
        assert 4 + int(3 * 4.6) <= opt._pop_size <= 4 + int(3 * 4.7) + 5

    def test_handles_nan_population_members(self, key):
        call_count = [0]

        def sometimes_nan(x):
            call_count[0] += 1
            return jnp.where(call_count[0] % 3 == 0, jnp.nan, jnp.sum(x**2))

        opt = EvolutionaryOptimizer(population_size=16, n_generations=10)
        state = opt.init(key, jnp.zeros(10), sphere)
        for _ in range(10):
            if opt.should_stop(state):
                break
            key, subkey = jax.random.split(key)
            state, _ = opt.step(state, subkey, sometimes_nan)
        assert jnp.isfinite(state.best_loss)

    def test_metrics_contain_sigma(self, key):
        opt = EvolutionaryOptimizer(population_size=16, n_generations=10)
        _, metrics = opt.step(opt.init(key, jnp.zeros(10), sphere), key, sphere)
        assert all(k in metrics for k in ("sigma", "gen_best_loss", "n_valid"))


class TestHybridOptimizer:
    def test_init_starts_in_ec_phase(self, key):
        hybrid = HybridOptimizer(
            ec=EvolutionaryOptimizer(n_generations=10),
            gd=GradientDescentOptimizer(n_steps=20),
            ec_generations=10,
        )
        assert hybrid.init(key, jnp.ones(10), sphere).phase == OptimPhase.EC

    def test_phase_transition_occurs(self, key):
        hybrid = HybridOptimizer(
            ec=EvolutionaryOptimizer(population_size=8, n_generations=10),
            gd=GradientDescentOptimizer(n_steps=20),
            ec_generations=5,
        )
        state = hybrid.init(key, jnp.ones(10), sphere)
        phases_seen = set()
        for _ in range(30):
            if hybrid.should_stop(state):
                break
            key, subkey = jax.random.split(key)
            state, metrics = hybrid.step(state, subkey, sphere)
            phases_seen.add(int(metrics["phase"]))
        assert OptimPhase.EC in phases_seen and OptimPhase.GD in phases_seen

    def test_handoff_preserves_best(self, key):
        hybrid = HybridOptimizer(
            ec=EvolutionaryOptimizer(population_size=8, n_generations=10),
            gd=GradientDescentOptimizer(n_steps=5),
            ec_generations=5,
        )
        state = hybrid.init(key, jnp.zeros(5), sphere)
        ec_best_loss = None
        for _ in range(20):
            key, subkey = jax.random.split(key)
            state, metrics = hybrid.step(state, subkey, sphere)
            if metrics.get("handoff"):
                ec_best_loss = float(metrics["ec_final_loss"])
                break
        assert ec_best_loss is not None and float(state.best_loss) == ec_best_loss

    def test_gd_improves_after_handoff(self, key):
        hybrid = HybridOptimizer(
            ec=EvolutionaryOptimizer(population_size=16, n_generations=10, sigma_init=0.5),
            gd=GradientDescentOptimizer(n_steps=100),
            ec_generations=10,
        )
        state = hybrid.init(key, jnp.ones(10) * 2, sphere)
        loss_at_handoff = None
        for _ in range(150):
            if hybrid.should_stop(state):
                break
            key, subkey = jax.random.split(key)
            state, metrics = hybrid.step(state, subkey, sphere)
            if metrics.get("handoff"):
                loss_at_handoff = float(state.best_loss)
        assert loss_at_handoff is not None and float(state.best_loss) < loss_at_handoff


class TestJITCompatibility:
    def test_gd_step_jit_compatible(self, key):
        opt = GradientDescentOptimizer(n_steps=10)
        state = opt.init(key, jnp.ones(10), sphere)
        jit_step = jax.jit(lambda s, k: opt.step(s, k, sphere))
        for _ in range(5):
            key, subkey = jax.random.split(key)
            state, _ = jit_step(state, subkey)
        assert state.step > 0

    def test_ec_step_jit_compatible(self, key):
        opt = EvolutionaryOptimizer(population_size=8, n_generations=10)
        state = opt.init(key, jnp.zeros(10), sphere)
        jit_step = jax.jit(lambda s, k: opt.step(s, k, sphere))
        for _ in range(5):
            key, subkey = jax.random.split(key)
            state, _ = jit_step(state, subkey)
        assert state.step > 0
