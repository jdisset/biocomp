from jax.experimental import checkify
import jax
from jax import jit
from jax import tree_util as jtu
import jax.numpy as jnp
import numpy as np


### {{{                        --     jax version of log-poly-log functions     --


def jlogb(x, base=10):
    """Compute log of x in base b."""
    return jnp.log(x) / jnp.log(base)


def jcubic_exp_fwd(x, threshold, base, scale=1):
    """
    cubic polynomial that goes through (0,0) and has same first
    and second derivative as the log function at the threshold
    In other works, a spline that is log-like near the threshold
    scale is a parameter to squeeze or stretch the function
    """
    # assert base > 1 and scale > 0, 'Base must be > 1 and scale > 0'
    # assert (
    # 6 * logb(threshold, base) * scale > 5
    # ), 'Threshold too small for given scale (or vice versa)'

    logthresh = jnp.log(threshold)
    logbase = jnp.log(base)
    a = -0.5 * (3 - 2 * scale * logthresh) / (threshold**3 * logbase)
    b = -(-4 + 3 * scale * logthresh) / (threshold**2 * logbase)
    c = -0.5 * (5 - 6 * scale * logthresh) / (threshold * logbase)
    return a * x**3 + b * x**2 + c * x


def jcubic_exp_inv(y, threshold, base, scale):
    """
    inverse of cubic_exp_fwd (on [0,threshold])
    """
    # used wolfram to solve the analytical inverse
    lT, lB, cb2 = jnp.log(threshold), jnp.log(base), jnp.cbrt(2)
    T, T2, T3 = threshold, threshold**2, threshold**3
    A = T3 * (
        56
        + y * lB * (486 - 648 * scale * lT + 216 * scale**2 * lT**2)
        - 522 * scale * lT
        + 648 * scale**2 * lT**2
        - 216 * scale**3 * lT**3
    )
    B = jnp.sqrt(4 * (-19 * T2 + 12 * scale * T2 * lT) ** 3 + A**2)
    C = jnp.cbrt(A + B)
    D = -9 + 6 * scale * lT
    E = 2 * T * (-4 + 3 * scale * lT) / D
    F = cb2 * (-19 * T2 + 12 * scale * T2 * lT)
    return E - (F / (D * C)) + (C / (cb2 * D))


@jit
def jax_log_poly_log(x, threshold=100, base=10, compression=0.5):
    """
    bi-logarithm function with smooth transition to cubic polynomial between [-threshold, threshold]
    """
    x = jnp.asarray(x)
    sign = jnp.sign(x)
    x = jnp.abs(x)
    diff = jlogb(threshold, base) * (1.0 - compression)
    x = jnp.where(
        x > threshold,
        jlogb(x, base) - diff,
        jcubic_exp_fwd(x, threshold, base=base, scale=compression),
    )
    return x * sign


@jit
def jax_inverse_log_poly_log(y, threshold=100, base=10, compression=0.5):
    y = jnp.asarray(y)
    sign = jnp.sign(y)
    y = jnp.abs(y)
    diff = jlogb(threshold, base) * (1.0 - compression)
    transformed_threshold = jcubic_exp_fwd(threshold, threshold, base=base, scale=compression)
    y = jnp.where(
        y > transformed_threshold,
        base ** (y + diff),
        jcubic_exp_inv(y, threshold, base=base, scale=compression),
    )
    return y * sign


##────────────────────────────────────────────────────────────────────────────}}}

## {{{                         --     misc utils     --

enable_checks = False


def get_jaxpr(fun, *args, **kwargs):
    import jax

    return jax.make_jaxpr(fun)(*args, **kwargs)


def print_jaxpr(fun, *args, **kwargs):
    get_jaxpr(fun, *args, **kwargs).pretty_print()


def get_xla(fun, *args, static_argnums=(), **kwargs):
    import jax
    import jaxlib.xla_extension as xla_ext
    from rich.console import Console
    import rich

    _console = Console(highlighter=rich.highlighter.ReprHighlighter())  # noqa: F841
    c = jax.xla_computation(fun, static_argnums=static_argnums)(*args, **kwargs)
    backend = jax.lib.xla_bridge.get_backend()
    e = backend.compile(c)
    option = xla_ext.HloPrintOptions.short_parsable()
    out = e.hlo_modules()[0].to_string(option)
    return out


def print_xla(fun, *args, static_argnums=(), **kwargs):
    print(get_xla(fun, *args, **kwargs))


def get_looped_slice(a, start, end, axis=0):
    """Get a slice of an array that loops around the end of the array if end > a.shape[axis]"""
    ns = a.__array_namespace__()  # grab numpy/jax namespace
    idx = ns.arange(start, end) % a.shape[axis]
    return a.take(idx, axis=axis)


def value_and_jacrev(f, x):
    y, pullback = jax.vjp(f, x)
    basis = jnp.eye(y.size, dtype=y.dtype)
    jac = jax.vmap(pullback)(basis)
    return y, jac


def freeze(struct):
    # converts dict to frozendict, list to tuple and recursively
    # freezes all nested dicts, lists, tuples, and sets.
    import frozendict

    if isinstance(struct, dict):
        return frozendict.frozendict({k: freeze(v) for k, v in struct.items()})
    elif isinstance(struct, list):
        return tuple([freeze(v) for v in struct])
    elif isinstance(struct, tuple):
        return tuple([freeze(v) for v in struct])
    elif isinstance(struct, set):
        return frozenset([freeze(v) for v in struct])
    else:
        return struct


def robust_sort(x, axis=-1):
    """
    a sort operation that's more stable under JAX transformations like checkify.

    """

    def _sort_1d(arr):
        perm = jnp.argsort(arr)
        one_hot = jax.nn.one_hot(perm, num_classes=arr.shape[0])
        return jnp.dot(one_hot, arr)

    x_swapped = jnp.moveaxis(x, axis, -1)

    sort_fn = _sort_1d
    for _ in range(x.ndim - 1):
        sort_fn = jax.vmap(sort_fn)

    sorted_swapped = sort_fn(x_swapped)

    return jnp.moveaxis(sorted_swapped, -1, axis)


def tree_shape(t):
    return jtu.tree_map(lambda x: x.shape, t)


@jit
def tree_append(t, e):
    fa, tt = jtu.tree_flatten(t)
    fb, te = jtu.tree_flatten(e)
    assert te == tt
    return jtu.tree_unflatten(tt, [jnp.concatenate([a, jnp.array([b])]) for a, b in zip(fa, fb)])


def tree_get(t, i):
    return jtu.tree_map(lambda x: x[i], t)


@jax.jit
def tree_unstack(t):
    """Unstack a tree of arrays into a list of trees of arrays"""
    N = jax.tree_util.tree_leaves(t)[0].shape[0]
    return [tree_get(t, i) for i in range(N)]


def set_enable_checks(value: bool):
    global enable_checks
    enable_checks = value


def check(*args, **kwargs):
    global enable_checks
    if enable_checks:
        checkify.check(*args, **kwargs)
    else:
        # replace by an assert of the same thing
        assert args[0](*args[1:], **kwargs)


def checkwrap(func, errors=(checkify.user_checks | checkify.index_checks | checkify.float_checks)):
    from jax.experimental.checkify import Error

    global enable_checks
    if enable_checks:
        return jit(checkify.checkify(func, errors=errors))
    else:

        def wrapped_function(*args, **kwargs):
            result = func(*args, **kwargs)
            return Error({}, {}, {}, {}), result

        return wrapped_function


def flat_concat(*arrays):
    return jnp.concatenate([jnp.asarray(a).ravel() for a in arrays])


def tree_to_jax(params):
    return jtu.tree_map(lambda x: jnp.asarray(x), params)


def tree_to_np(params):
    return jtu.tree_map(lambda x: np.asarray(x), params)


##────────────────────────────────────────────────────────────────────────────}}}
