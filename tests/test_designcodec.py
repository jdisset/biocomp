import pytest
import jax.numpy as jnp

from biocomp.designcodec import GenomeCodec
from biocomp.parameters import ParameterTree


@pytest.fixture
def simple_params():
    params = ParameterTree()
    params.at("layer1/weights", jnp.ones((4, 3)))
    params.at("layer1/bias", jnp.zeros(4))
    params.at("layer2/weights", jnp.ones((2, 4)) * 0.5)
    return params


@pytest.fixture
def params_with_tags():
    params = ParameterTree()
    params.at("shared/global_scale", jnp.array([1.0]), tags=["shared"])
    params.at("layer1/weights", jnp.ones((4, 3)))
    params.at("layer1/bias", jnp.zeros(4))
    return params


@pytest.fixture
def codec(simple_params):
    return GenomeCodec.from_params(simple_params)


class TestCodecCreation:
    def test_from_params_sets_param_dim(self, simple_params):
        codec = GenomeCodec.from_params(simple_params)
        assert codec.param_dim == 4 * 3 + 4 + 2 * 4

    def test_from_params_with_static_tags(self, params_with_tags):
        codec = GenomeCodec.from_params(params_with_tags, static_tags=("shared",))
        assert codec.param_dim == 4 * 3 + 4

    def test_static_params_stored(self, params_with_tags):
        codec = GenomeCodec.from_params(params_with_tags, static_tags=("shared",))
        assert codec.static_params.data is not None


class TestCodecRoundtrip:
    def test_encode_decode_preserves_shape(self, codec, simple_params):
        flat = codec.encode(simple_params)
        decoded = codec.decode(flat, apply_constraints=False)
        orig = sorted([(str(p), v) for p, v in simple_params.data.iter_leaves()])
        dec = sorted([(str(p), v) for p, v in decoded.data.iter_leaves()])
        for (p1, v1), (_, v2) in zip(orig, dec):
            assert v1.shape == v2.shape, f"shape mismatch at {p1}"

    def test_encode_decode_preserves_values(self, codec, simple_params):
        flat = codec.encode(simple_params)
        decoded = codec.decode(flat, apply_constraints=False)
        orig = sorted([(str(p), v) for p, v in simple_params.data.iter_leaves()])
        dec = sorted([(str(p), v) for p, v in decoded.data.iter_leaves()])
        for (p1, v1), (_, v2) in zip(orig, dec):
            assert jnp.allclose(v1, v2, rtol=1e-5), f"value mismatch at {p1}"

    def test_roundtrip_dimension_matches(self, codec, simple_params):
        assert codec.encode(simple_params).shape[0] == codec.param_dim


class TestCodecEncode:
    def test_encode_returns_flat_vector(self, codec, simple_params):
        assert codec.encode(simple_params).ndim == 1

    def test_encode_wrong_structure_raises(self, codec):
        wrong = ParameterTree()
        wrong.at("wrong/path", jnp.ones(10))
        with pytest.raises(AssertionError):
            codec.encode(wrong)


class TestCodecDecode:
    def test_decode_wrong_dimension_raises(self, codec):
        with pytest.raises(AssertionError, match="dim"):
            codec.decode(jnp.ones(codec.param_dim + 10))

    def test_decode_with_constraints(self, codec):
        assert codec.decode(jnp.ones(codec.param_dim) * 1000.0, apply_constraints=True) is not None


class TestCodecValidation:
    def test_validate_genome_nan(self, codec):
        from jax.experimental import checkify

        nan_genome = jnp.ones(codec.param_dim).at[0].set(jnp.nan)
        err, _ = checkify.checkify(codec.validate_genome, errors=checkify.user_checks)(nan_genome)
        with pytest.raises(Exception):
            err.throw()

    def test_validate_genome_wrong_dim(self, codec):
        from jax.experimental import checkify

        err, _ = checkify.checkify(codec.validate_genome, errors=checkify.user_checks)(
            jnp.ones(codec.param_dim + 5)
        )
        with pytest.raises(Exception):
            err.throw()


class TestCodecBounds:
    def test_bounds_returns_correct_shape(self, codec):
        lower, upper = codec.bounds()
        assert lower.shape == upper.shape == (codec.param_dim,)

    def test_bounds_default_infinite(self, codec):
        lower, upper = codec.bounds()
        assert jnp.all(lower == -jnp.inf) and jnp.all(upper == jnp.inf)
