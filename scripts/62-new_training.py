from biocomp import utils as ut
import numpy as np
import scriptutils as su
import optax
import time
from copy import deepcopy
import biocomp.datautils as du
import biocomp.train as train
import biocomp.parameters as pm
import biocomp.nodes as nd
import biocomp.compute as cmp
import biocomp
import jax
from jax import jit, grad, vmap, random, value_and_grad
from jax import numpy as jnp
import jax.tree_util as jtu
import cProfile

class profiler:
    def __init__(self, filename):
        self.filename = filename
    def __enter__(self):
        self.profiler = cProfile.Profile()
        self.profiler.enable()
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.profiler.disable()
        self.profiler.dump_stats(self.filename)

prog = train.TrainingProgram()
prog.parse_args()
ut.set_loglevel('info')


### {{{                      --     loading xp     --

XP = {
    'bt': '2023-04-03_Constraints_Pgu_Bleedthrough',
    'cascades': '2023-04-18_Constraints_PguCascades',
    'csy4matrix': '2023-03-26_MatrixCsy4',
    'casematrix': '2023-02-16_Matrix',
}
# xpnames = ['bt', 'cascades', 'csy4matrix', 'casematrix']
xpnames = XP.keys()


with ut.timer(f'Loading data and building networks for {xpnames}'):
    lib = su.load_lib()
    loadedxp = {
        xpname: su.load_xp(XP[xpname], lib, data_path='./data/calibrated_data_v2')
        for xpname in xpnames
    }

    dman_full = du.DataManager.from_xps(loadedxp.values(), prog.training_config, inverse='all')

all_networks = dman_full.get_networks()
net_xp = [n.metadata['from_xp'] for n in all_networks]
net_name = [n.name for n in all_networks]

##────────────────────────────────────────────────────────────────────────────}}}

### {{{               --     training and validation sets     --

# list net names that have cascade in the name:
inert_nets = {n: i for i, n in enumerate(net_name) if 'inert' in n.lower()}
cascade_nets = {
    n: i for i, n in enumerate(net_name) if 'cascade' in n.lower() and 'inert' not in n.lower()
}

# training set is all networks except the ones in inert or cascade
training_set = [
    i
    for i, _ in enumerate(net_name)
    if i not in inert_nets.values() and i not in cascade_nets.values()
]

validation_set = [
    i for i, _ in enumerate(net_name) if i not in inert_nets.values() and i in cascade_nets.values()
]

n_outputs = [n.get_nb_outputs() for n in all_networks]

validation = dman_full.make_subset(validation_set)
training = dman_full.make_subset(training_set)



# prog.start_training(dman_full.make_subset(training_set), validation)

##────────────────────────────────────────────────────────────────────────────}}}

compute_config = cmp.DEFAULT_COMPUTE_CONFIG
key = jax.random.PRNGKey(0)

xbatches, ybatches = training.get_batches(key)

##

compute_config = cmp.DEFAULT_COMPUTE_CONFIG
key = jax.random.PRNGKey(0)
seed = 42
training_config = deepcopy(biocomp.train.DEFAULT_TRAINING_CONFIG)
dman = training

ut.logger.debug(f"About to start training")
ut.logger.debug(f"Training config: {training_config}")
ut.logger.debug(f"Compute config: {compute_config.config}")

if seed is not None:
    training_config['rng_key'] = seed

ut.logger.info(f"Going to train with random seed {training_config['rng_key']}")
key = jax.random.PRNGKey(training_config['rng_key'])

def init_stack(dman, key):
    stack = dman.build_compute_stack(compute_config)
    with ut.timer('Stack initialization'):
        params = stack.init(key)
    return stack, params

def generate_batches(dman, key):
    with ut.timer('Generating batches'):
        xbatches, ybatches = dman.get_batches(key)  # (B,M,N,F) shape
    return xbatches, ybatches

stack, params = init_stack(dman, key)
xbatches, ybatches = generate_batches(dman, key)

ut.logger.info(f"Generated {xbatches.shape[0]} batches")
optimizer = biocomp.train.get_optimizer(training_config)


static, dynamic = params.filter_by_tag(['non_grad', 'local'])


ut.logger.info(f"Split params between dynamic and static. Now intializing optimizer.")

opt_state = optimizer.init(dynamic)
total_batches = training_config['n_batches']
assert total_batches == xbatches.shape[0] == ybatches.shape[0]
steps_per_epoch = max(1, int(training_config['steps_per_epoch']))
ut.logger.info(f"Done initializing optimizer, total batches: {total_batches}, steps per epoch: {steps_per_epoch}")

# --- loss & update functions

vmapped_compute = jax.vmap(stack.apply, in_axes=(None, 0, 0, 0))

def loss_func(dynamic, static, X, Y, Z, key):
    nb_inputs = sum([n.get_nb_inputs() for n in stack.networks])
    nb_outputs = sum([n.get_nb_outputs() for n in stack.networks])
    assert X.ndim == Y.ndim == Z.ndim == 2, "X, Y, and Z must have 2 dimensions"
    assert (
        X.shape[0] == Y.shape[0] == Z.shape[0]
    ), "X, Y, and Z must have the same number of rows"
    assert (
        X.shape[1] == nb_inputs
    ), "X must have as many columns as the total number of inputs in the stack"
    assert (
        Y.shape[1] == Z.shape[1] == nb_outputs
    ), "Y and Z must have as many columns as the total number of outputs in the stack"

    # params = ut.assemble_params(dynamic, static)
    params = pm.ParameterTree.merge(dynamic, static)
    keys = jax.random.split(key, X.shape[0])

    yhat, grads = vmapped_compute(params, X, Z, keys)
    assert yhat.shape == Y.shape, "yhat and Y must have the same shape"

    error = yhat - Y
    quantile_loss = jnp.mean(
        biocomp.train.huber_quantile_loss(error, Z, delta=training_config['huber_quantile_loss_delta'])
    )

    # grads is the concatenated and flattened jacobian of
    # translate, transcript, and output nodes wrt their inputs
    # they should be monotonically increasing so we add a loss term
    negative_grads = jnp.mean(jnp.where(grads < 0, -grads, 0))
    return quantile_loss + training_config['negative_grad_penalty'] * negative_grads

@jit
def training_step(params, opt_state, x, y, z, key):
    static, dynamic = params.filter_by_tag(['non_grad', 'local'])
    loss, grads = value_and_grad(loss_func, has_aux=False)(dynamic, static, x, y, z, key)
    updates, opt_state = optimizer.update(grads, opt_state, dynamic)
    # dynamic = optax.apply_updates(dynamic, updates)
    params = pm.ParameterTree.merge(static, dynamic)
    res = {
        'params': params,
        'loss': loss,
        'grad': grads,
        'opt': opt_state,
    }
    return res



##
keep_in_history = training_config.get('keep_in_history', ['loss'])

def scannable_step(carry, i_x_y_z_k):
    params, opt_state = carry
    i, x, y, z, k = i_x_y_z_k
    updt = training_step(params, opt_state, x, y, z, k)
    params, opt_state = updt['params'], updt['opt']
    history = {k: updt[k] for k in keep_in_history}
    return (params, opt_state), history

def epoch_step(start_params, start_opt_state, epoch_key, xbs, ybs):
    pscan = ut.progress_scan(steps_per_epoch, message='Training model')
    zbatches = jax.random.uniform(epoch_key, ybs.shape)
    batch_keys = jax.random.split(epoch_key, steps_per_epoch)
    sstep = pscan(scannable_step)
    (final_params, final_opt_state), epoch_history = jax.lax.scan(
        sstep,
        (start_params, start_opt_state),
        (jnp.arange(steps_per_epoch), xbs, ybs, zbatches, batch_keys),
    )
    return final_params, final_opt_state, epoch_history

def epoch_step_no_scan(start_params, start_opt_state, epoch_key, xbs, ybs):
    zbatches = jax.random.uniform(epoch_key, ybs.shape)
    batch_keys = jax.random.split(epoch_key, steps_per_epoch)
    all_history = []
    tstep = training_step
    for i, (x, y, z, k) in tqdm(
        enumerate(zip(xbs, ybs, zbatches, batch_keys)), total=steps_per_epoch
    ):
        updt = tstep(start_params, start_opt_state, x, y, z, k)
        start_params, start_opt_state = updt['params'], updt['opt']
        history = {k: updt[k] for k in keep_in_history}
        all_history.append(history)
    epoch_history = {k: jnp.stack([h[k] for h in all_history]) for k in keep_in_history}
    return start_params, start_opt_state, epoch_history

epoch_step = epoch_step if not ut.enable_checks else epoch_step_no_scan

jax.clear_caches()

print(compute_config.dumps())
##

with ut.timer('lowering the epoch_step function'):
    xb = ut.get_looped_slice(xbatches, 0 * steps_per_epoch, steps_per_epoch)
    yb = ut.get_looped_slice(ybatches, 0 * steps_per_epoch, steps_per_epoch)
    lowered = jax.jit(epoch_step).lower(params, opt_state, key, xb, yb)

##

with ut.timer('Compiling the epoch_step function'):
    compiled_epoch_step = lowered.compile()


##
# --- main training loop

loggers = [(1, biocomp.train.console_log)]

for _, l in loggers:
    l(epoch=0, training_config=training_config)

ut.logger.info(f'Begin training for {training_config["epochs"]} epochs')

training_config['epochs'] = 1

for i, epoch_key in enumerate(jax.random.split(key, training_config['epochs']), 1):

    t0 = time.time()
    xb = ut.get_looped_slice(xbatches, i * steps_per_epoch, (i + 1) * steps_per_epoch)
    yb = ut.get_looped_slice(ybatches, i * steps_per_epoch, (i + 1) * steps_per_epoch)
    params, opt_state, epoch_history = compiled_epoch_step(params, opt_state, epoch_key, xb, yb)
    epoch_history['epoch_time'] = time.time() - t0
    epoch_history['latest_params'] = params

    for t, l in loggers:
        if t is not None:
            if (t == 0 or (i % t == 0 and t > 0)) or i == training_config['epochs']:
                l(
                    epoch=i,
                    training_config=training_config,
                    epoch_history=epoch_history,
                    nbatches=steps_per_epoch,
                )


print('done')

##

