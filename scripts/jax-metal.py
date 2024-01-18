# simple sanity check for jax-metal: let's start by inverting a matrix
import jax
import numpy as np
import jax.numpy as jnp

# basic
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
import timeit



def selu(x, alpha=1.67, lmbda=1.05):
  return lmbda * jnp.where(x > 0, x, alpha * jnp.exp(x) - alpha)

key = random.PRNGKey(0)
x = random.normal(key, (10,))

timeit.timeit(lambda: selu(x).block_until_ready(), number=1000)
selu_jit = jit(selu)
timeit.timeit(lambda: selu_jit(x).block_until_ready(), number=1000)


def sum_logistic(x):
  return jnp.sum(1.0 / (1.0 + jnp.exp(-x)))

x_small = jnp.arange(3.)
derivative_fn = grad(sum_logistic)
print(derivative_fn(x_small))
def first_finite_differences(f, x):
  eps = 1e-3
  return jnp.array([(f(x + eps * v) - f(x - eps * v)) / (2 * eps)
                   for v in jnp.eye(len(x))])

print(first_finite_differences(sum_logistic, x_small))
print(grad(jit(grad(jit(grad(sum_logistic)))))(1.0))
from jax import jacfwd, jacrev
def hessian(fun):
  return jit(jacfwd(jacrev(fun)))

mat = random.normal(key, (150, 100))
batched_x = random.normal(key, (10, 100))

def apply_matrix(v):
  return jnp.dot(mat, v)

def naively_batched_apply_matrix(v_batched):
  return jnp.stack([apply_matrix(v) for v in v_batched])

print('Naively batched')
timeit.timeit(lambda: naively_batched_apply_matrix(batched_x).block_until_ready(), number=10)

@jit
def batched_apply_matrix(v_batched):
  return jnp.dot(v_batched, mat.T)

print('Manually batched')
timeit.timeit(lambda: batched_apply_matrix(batched_x).block_until_ready(), number=10)

@jit
def vmap_batched_apply_matrix(v_batched):
  return vmap(apply_matrix)(v_batched)

print('Auto-vectorized with vmap')
timeit.timeit(lambda: vmap_batched_apply_matrix(batched_x).block_until_ready(), number=10)



##


# simple linalg

A = np.random.randn(100, 100)
Ainv = np.linalg.inv(A)
Ainv_jax = jnp.linalg.inv(A)
assert np.allclose(Ainv, Ainv_jax)
