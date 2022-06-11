import jax
from jax.tree_util import Partial as partial
import jax.numpy as jnp
from jax.example_libraries import stax
from jax.example_libraries.stax import Dense, Relu
from rich import print


def backends():
  backends = []
  for backend in ['cpu', 'gpu', 'tpu']:
    try:
      jax.devices(backend)
    except RuntimeError:
      pass
    else:
      backends.append(backend)
  return backends

print('Available backends:', backends(), ' | default:', jax.default_backend())

key = jax.random.PRNGKey(1)
init_fun, model = stax.serial(Dense(16), Relu, Dense(1))
model = jax.jit(model)
_, params = init_fun(key, (4,))

inputs = jax.random.uniform(key=key, shape=(1024, 4))
print(inputs.device_buffer.device())
print('computed',jax.vmap(partial(model,params))(inputs).mean())
