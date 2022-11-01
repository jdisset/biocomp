import biocomp as bc
import biocomp.compute as bcc
import numpy as np
from functools import partial
import biocomp.utils as bu
import scriptutils as ut
import jax
import jax.numpy as jnp
import random
random.seed()

lib = ut.load_lib()
xp = ut.load_xp('20221012A_massCtrls', lib)

cfg = {
    "learning_rate": 0.001,
    "node_remap": {"sequestron_ERN": "ERN_with_affinity"},
    "rng_key": random.randint(0, 1e9),
}
##

models = xp.get_models()
# model is a dict. Let's get one

model = models.items().__iter__().__next__()[1]

p = model.init(jax.random.PRNGKey(cfg['rng_key']))

inp = jnp.array([1.0, 2.0])
model.apply(p, inp, rng_key=jax.random.PRNGKey(0))
model.collect_all_results(p, inp, rng_key=jax.random.PRNGKey(0))


# lowered = jax.jit(model.apply).lower(p, inp, rng_key=jax.random.PRNGKey(0))
# compiled = lowered.compile()
# print(compiled.as_text())
# compiled.cost_analysis()
# compiled.memory_analysis()
# compiled(p, inp, rng_key=jax.random.PRNGKey(0))

# bc.train.train_xp(xp, cfg, wandb_project="biocomp_20221012A_massCtrls")
##


## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                        --     helpers   --
#···············································································
def glorot_initializer(rng_key, shape):
    def init():
        return jax.nn.initializers.glorot_normal()(rng_key, shape)
    return init

params = {}
get_p = partial(
    bcc.get_param,
    params,
    node_id=1,
)
get_q = partial(
    bcc.get_quantized,
    params,
    quantize_fun=bcc.quantize,
    node_id=1,
)


#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────


def nn_dense(input_values, hidden_size, output_size, get_param, key, name):
    k1, k2 = jax.random.split(key, 2)
    input_size = input_values.shape[-1]
    w1 = get_param(f'{name}_w1', init=glorot_initializer(k1, (input_size, hidden_size)), shared=True)
    b1 = get_param(f'{name}_b1', init=lambda: jnp.zeros((hidden_size,)), shared=True)
    w2 = get_param(f'{name}_w2', init=glorot_initializer(k2, (hidden_size, output_size)), shared=True)
    b2 = get_param(f'{name}_b2', init=lambda: jnp.zeros((output_size,)), shared=True)
    return jnp.dot(jax.nn.sigmoid(jnp.dot(input_values, w1) + b1), w2) + b2


y = nn_dense(jnp.array([1.0]), 2, get_p, jax.random.PRNGKey(1), 'test')

# def nn_dense_multilevel(input_values, hidden_size, output_size, depth, get_param, key, name):
    # # similar to n_dense, but instead of a single layer, we have a stack of layers (depth)
    # # each layer is a dense layer, but the input to each layer is the output of the previous layer
    # res = input_values
    # keys = jax.random.split(key, depth)
    # for i in range(depth - 1):
        # res = nn_dense(res, hidden_size, hidden_size, get_param, keys[i], f'{name}_{i}')
    # return nn_dense(res, hidden_size, output_size, get_param, keys[-1], f'{name}_{depth - 1}')

# def transform_w_dense_layer(get_param, get_quantized, wsize, transform_name, deg_param_name, **_):
    # app = bcc._transform(get_param, get_quantized, transform_name, deg_param_name, **_)
    # def apply(*values, rng_key):
        # k1, k2 = jax.random.split(rng_key, 2)
        # res = app(*values, rng_key=k1)
        # return nn_dense(res, wsize, get_param, k2, transform_name)
    # return apply

##

def transform_hill(get_param, get_quantized, transform_name, specie_name, **_):
    def apply(*values, rng_key):
        keys = jax.random.split(rng_key, 4)

        #"$$ [out] = V_{specie} (\\frac{[value]}{[value] + K_{A_{specie}}})^{n_{specie}} / (\\mu_{specie}) $$",

        V_specie_name = f'V_{specie_name}'

        V_species_raw = get_quantized(
                    V_specie_name,
                    get_param(V_specie_name, init=bcc.continuous_initializer(keys[0], len(values))),
                    mode='input_edges',
                ).squeeze()
        values = jnp.array(values).squeeze()

        # we compute the actual V_species as a weighted average of the V_species_raw 
        # (which depend on input edges (i.e. which promoter or uORF is used))
        V = jnp.average(V_species_raw, weights=values)

        K_A= get_param(f'{transform_name}_K_A_{specie_name}', init=bcc.continuous_initializer(keys[1]), shared=True)
        n= get_param(f'{transform_name}_n_{specie_name}', init=bcc.continuous_initializer(keys[2]), shared=True)
        mu= get_param(f'{transform_name}_mu_{specie_name}', init=bcc.continuous_initializer(keys[3]), shared=True)

        value = jnp.sum(values)

        return V * (value / (value + K_A)) ** n / mu


    return apply


def inverse_transform_hill(get_param, get_quantized, transform_name, specie_name, **_):
    def apply(values, rng_key):
        keys = jax.random.split(rng_key, 4)
        assert len(values) == 1

        # [out]^* = \\frac{\\alpha K_{A_{specie}}}{1 - \\alpha}
        # where $\\alpha$ is:\n",
        # \\alpha = \\sqrt[n_{specie}]{\\frac{[in]}{V_{specie}} (\\mu_{specie})}

        V_specie_name = f'V_{specie_name}'
        V_specie = get_quantized(
                    V_specie_name,
                    get_param(V_specie_name, init=bcc.continuous_initializer(keys[0], len(values))),
                    mode='input_edges',
                )

        assert len(V_specie) == 1

        V = V_specie[0]
        val = values[0]

        K_A = get_param(f'{transform_name}_K_A_{specie_name}', init=bcc.continuous_initializer(keys[1]), shared=True)
        n = get_param(f'{transform_name}_n_{specie_name}', init=bcc.continuous_initializer(keys[2]), shared=True)
        mu = get_param(f'{transform_name}_mu_{specie_name}', init=bcc.continuous_initializer(keys[3]), shared=True)

        alpha = jnp.power(val / V * mu, 1 / n)
        return jnp.array([alpha * K_A / (1 - alpha)])

    return apply

def get_qu(_, v, **__):
    return v

f = transform_hill(get_p, get_qu, 'test', 'A')
inv_f = inverse_transform_hill(get_p, get_qu, 'test', 'A')

y = f(jnp.array([0.05322093000]), rng_key=jax.random.PRNGKey(1))
y.shape
inv_f(y, rng_key=jax.random.PRNGKey(1))









