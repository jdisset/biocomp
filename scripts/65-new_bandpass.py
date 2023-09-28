### {{{                          --     imports     --
from biocomp import utils as ut
import scriptutils as su
import biocomp.datautils as du
import time
import biocomp.train as train
import biocomp.compute as cmp
import biocomp.parameters as pm
import biocomp as bc
from biocomp.parameters import ParameterTree
from jax.tree_util import Partial as partial
import jax.tree_util as jtu
from pathlib import Path
import jax.numpy as jnp
from copy import deepcopy
import optax
from tqdm import tqdm

import numpy as np
import jax
from jax import jit, grad, vmap, random, value_and_grad
from jax import numpy as jnp

from matplotlib import pyplot as plt

##────────────────────────────────────────────────────────────────────────────}}}

### {{{                     --     generate networks     --

lib = su.load_lib()


def any_uorf(lib, *_, **__):
    all_uORFs = lib.pc[lib.pc.category == 'uORF_group'].index.tolist()
    return [all_uORFs]


def P(name):
    return bc.Slot(lib, name)


def TU(*parts):
    partlist = [P('hEF1a')] + list(parts)
    return bc.TranscriptionUnit(partlist)


uorfs = P(any_uorf(lib)[0][:8])
# 'Csy4+uOrfs': bc.TranscriptionUnit([promoter, P('Csy4'), P(any_uorf(lib)[0])]),

ERNs = ['CasE', 'Csy4', 'PgU']
ern = [P(ern) for ern in ERNs]
rec = [P(ern + '_rec') for ern in ERNs]
biascolor = 'NeonGreen'
x0color_part = P('mKate')
x1color_part = P('eBFP')
outcolor_part = P('iRFP720')
biascolor_part = P(biascolor)
tus_bp = {
    # node A
    'A_pos_0': TU(rec[0], uorfs, ern[2]),
    'A_pos_1': TU(rec[0], uorfs, ern[2]),
    'A_pos_2': TU(rec[0], uorfs, ern[2]),
    'A_neg_0': TU(ern[0]),
    'A_neg_1': TU(ern[0]),
    'A_neg_2': TU(ern[0]),
    # node B
    'B_pos_0': TU(rec[1], uorfs, ern[2]),
    'B_pos_1': TU(rec[1], uorfs, ern[2]),
    'B_pos_2': TU(rec[1], uorfs, ern[2]),
    'B_neg_0': TU(ern[1]),
    'B_neg_1': TU(ern[1]),
    'B_neg_2': TU(ern[1]),
    # output node
    'C_pos_0': TU(rec[2], outcolor_part),
    'C_pos_1': TU(rec[2], outcolor_part),
    'C_pos_2': TU(rec[2], outcolor_part),
    'C_neg_0': TU(ern[2]),
    'C_neg_1': TU(ern[2]),
    'C_neg_2': TU(ern[2]),
    # colors
    'x0color': TU(x0color_part),
    'x1color': TU(x1color_part),
    'biascolor': TU(biascolor_part),
}



# everything everywhere all at once:
aggregations_bp = [
    ['A_pos_0', 'A_neg_0', 'B_pos_0', 'B_neg_0', 'C_pos_0', 'C_neg_0', 'x0color'],  # x0
    ['A_pos_1', 'A_neg_1', 'B_pos_1', 'B_neg_1', 'C_pos_1', 'C_neg_1', 'x1color'],  # x1
    ['A_pos_2', 'A_neg_2', 'B_pos_2', 'B_neg_2', 'C_pos_2', 'C_neg_2', 'biascolor'],  # bias
]


sources_bp = {
    tu_name: [tu_name] for tu_name, tu in tus_bp.items() if tu_name in ut.flatten(aggregations_bp)
}
used_tus_bp = {
    tu_name: tu for tu_name, tu in tus_bp.items() if tu_name in ut.flatten(aggregations_bp)
}

n_bp = bc.Network.from_dict(lib, 'bp_attempt', used_tus_bp, sources_bp, aggregations_bp)
bp_net = bc.inverted_network(n_bp)[0]

bp_net.set_input_as_bias(biascolor)


networks = [bp_net]

# dirname = Path('~/Desktop/bandpass_attempt/v0/networks/').expanduser()
# dirname.mkdir(parents=True, exist_ok=True)
# su.plot_networks(networks, filenames=[f'{dirname}/network_{i}.pdf' for i in range(len(networks))])
su.plot_networks(networks, W=4500, H=4000, show=True, figsize=(22, 20))

NETWORK = networks[0]


##────────────────────────────────────────────────────────────────────────────}}}

### {{{             --     generate random bandpass designs     --
# in logspace.


training_config = train.DEFAULT_TRAINING_CONFIG
logtr_conf = {
    'offset': training_config['data_log_offset'],
    'maxv': training_config['data_max_value'],
    'factor': training_config['data_log_factor'],
    'threshold': training_config['data_log_poly_threshold'],
    'compression': training_config['data_log_poly_compression'],
}

tr = partial(du.tr, **logtr_conf)
inv_tr = partial(du.inv_tr, **logtr_conf)

vlims = np.array([logtr_conf['offset'], logtr_conf['maxv']])

vlims_log = tr(vlims) * 0.90

vlims_log

vrange = vlims_log[1] - vlims_log[0]
on_value = 1.0
off_value = 0.0
pmargin = 0.1


def gen_unit(teta_min, teta_max, key):
    teta = jax.random.uniform(key, (1,), minval=teta_min, maxval=teta_max)[0]
    return jnp.array([jnp.cos(teta), jnp.sin(teta)])


def is_inside(x, p, n):
    return jnp.dot(x - p, n) < 0


def inside_all(x, pvec, nvec):
    return jnp.all(vmap(is_inside, in_axes=(None, 0, 0))(x, pvec, nvec), axis=0)


@jit
def gen_bandpass(key):
    k0, k1, k2, k3 = jax.random.split(key, 4)

    left_normals = gen_unit(-np.pi, np.pi, k0)

    right_normals = -left_normals + jax.random.normal(k1, (1, 2)) * 0.2

    left_points = jax.random.uniform(
        k2, (1, 2), minval=vlims_log[0] + vrange * pmargin, maxval=vlims_log[1] - vrange * pmargin
    )
    d = jax.random.uniform(k3, (1,), minval=0.1, maxval=0.85)[0]
    right_points = left_points - d * (vlims_log[1] - vlims_log[0]) * left_normals

    points = jnp.concatenate([left_points, right_points], axis=0)
    normals = jnp.vstack([left_normals, right_normals])
    return points, normals


@partial(jit, static_argnums=(2,))
def gen_bandpass_xz(vlims, key, nsamples=10000):
    k0, k1 = jax.random.split(key, 2)
    points, normals = gen_bandpass(k0)
    x = jax.random.uniform(k1, (nsamples, 2), minval=vlims[0], maxval=vlims[1])
    y = vmap(inside_all, in_axes=(0, None, None))(x, points, normals)
    y = jnp.where(y, on_value, off_value)
    y = y.reshape(-1, 1)
    return x, y


rng = jax.random.PRNGKey(3)
NBP = 25
bandpasses = [gen_bandpass_xz(vlims_log, k, nsamples=50000) for k in jax.random.split(rng, NBP)]

for i, (x, z) in enumerate(bandpasses):
    print(i)
    fig, ax = plt.subplots()
    ax.scatter(x[:, 0], x[:, 1], c=z, s=2, cmap='YlGnBu', vmin=0, vmax=0.6)
    ax.set_xlim(vlims_log)
    ax.set_ylim(vlims_log)
    plt.show()


##────────────────────────────────────────────────────────────────────────────}}}

### {{{                        --     train utils     --

def get_output_indices(stack, output_protein_name):
    out_indices = []
    for n_id, n in enumerate(stack.networks):
        output_id = n.get_output_proteins().index(output_protein_name)
        out_indices.append(stack.get_network_global_output_id(n_id, output_id))
    return jnp.array(out_indices)




def generate_batches(bandpasses, n_batches, batch_size, key):
    assert isinstance(bandpasses, (list, tuple))
    Xs, Ys = zip(*bandpasses)
    Xs = np.array(Xs).transpose(1, 2, 0)
    Ys = np.array(Ys).transpose(1, 2, 0)
    idxs = jax.random.choice(
        key, np.arange(Xs.shape[0]), shape=(n_batches, batch_size), replace=True
    )
    return Xs[idxs], Ys[idxs]


def loss_check_shapes(stack, X, Y, Z):
    nb_inputs = sum([n.get_nb_inputs() for n in stack.networks])
    nb_outputs = sum([n.get_nb_outputs() for n in stack.networks])
    assert X.ndim == Y.ndim == Z.ndim == 2, f'Xdim: {X.ndim}, Ydim: {Y.ndim}, Zdim: {Z.ndim}'
    assert X.shape[0] == Y.shape[0] == Z.shape[0], f'X: {X.shape}, Y: {Y.shape}, Z: {Z.shape}'
    assert (
        X.shape[1] == nb_inputs
    ), f'X.shape[1] ({X.shape[1]}) must be equal to the total number of inputs in the stack ({nb_inputs})'
    assert (
        Y.shape[1] == Z.shape[1] == nb_outputs
    ), f'Y.shape[1] ({Y.shape[1]}) and Z.shape[1] ({Z.shape[1]}) must be equal to the total number of outputs in the stack ({nb_outputs})'


##────────────────────────────────────────────────────────────────────────────}}}

### {{{                      --     load parameters     --
training_archive = du.load('../__results/training_archives/20230923_fulltrain_v0.pkl')
shared_parameters = training_archive['parameters']
compute_config = training_archive['compute_config']
training_config = training_archive['training_config']
compute_config.set_impl('bias', bc.nodes.bias)
##────────────────────────────────────────────────────────────────────────────}}}


from jax import config
config.update("jax_debug_nans", False)

seed = 10213
total_batches = 2000
batch_size = 50
loggers = None
BP = [bandpasses[3]]*50
bias_protein_names = ['NeonGreen']
output_protein_name = 'iRFP720'
if seed is not None:
    training_config['rng_key'] = seed
key = jax.random.PRNGKey(training_config['rng_key'])
training_config['steps_per_epoch'] = 100
training_config['epochs'] = 40

def init_stack(rng):
    local_params, _ = stack.init(rng).filter_by_tag('local')
    local_params.data.check()
    full_params = ParameterTree.merge(local_params, shared_parameters)
    return full_params

# --- init & batches generation

# generate the compute stack
stack = cmp.ComputeStack([NETWORK])
stack.build(compute_config)
output_indices = get_output_indices(stack, output_protein_name)


full_params = vmap(init_stack, out_axes=-1)(jax.random.split(key, len(BP)))
full_params.data.check()
static_params, dynamic_params = full_params.filter_by_tag(['shared', 'non_grad'], mode='any')
xbatches, ybatches = generate_batches(BP, total_batches, batch_size, key)
ut.logger.info(f"Generated {xbatches.shape[0]} batches")

optimizer = bc.train.get_optimizer(training_config)
opt_state = optimizer.init(dynamic_params)

assert total_batches == xbatches.shape[0] == ybatches.shape[0]
steps_per_epoch = max(1, int(training_config['steps_per_epoch']))
ut.logger.info(
    f"Done initializing optimizer, total batches: {total_batches}, steps per epoch: {steps_per_epoch}"
)

# --- loss & update functions

vmapped_compute = jax.vmap(stack.apply, in_axes=(None, 0, 0, 0))


def evaluate_at(params, X, Z, key):
    keys = jax.random.split(key, X.shape[0])
    full_yhat, _ = vmapped_compute(params, X, Z, keys)
    loss_check_shapes(stack, X, full_yhat, Z)
    yhat = full_yhat[:, output_indices]
    if yhat.ndim == 1:
        yhat = yhat.reshape(-1, 1)
    return yhat


def loss_func(dynamic, static, X, Y, Z, key):
    yhat = evaluate_at(ParameterTree.merge(dynamic, static), X, Z, key).squeeze()
    y_on = (Y > 0.5).squeeze()
    y_off = (Y < 0.5).squeeze()
    n_on = jnp.sum(y_on)
    n_off = jnp.sum(y_off)
    yhat_on_avg = jnp.sum(jnp.where(y_on, yhat, 0)) / jnp.maximum(n_on, 1)
    yhat_off_avg = jnp.sum(jnp.where(y_off, yhat, 0)) / jnp.maximum(n_off, 1)
    loss = yhat_off_avg - yhat_on_avg
    return loss


def vgloss(dynamic, static, X, Y, Z, key):
    l, g = jax.vmap(value_and_grad(loss_func), in_axes=(-1, -1, -1, -1, -1, 0), out_axes=-1)(dynamic, static, X, Y, Z, key)
    return l, g

def training_step(params, opt_state, x, y, key):
    static, dynamic = params.filter_by_tag(['shared', 'non_grad'], mode='any')
    z = jax.random.uniform(key, (x.shape[0], stack.total_nb_of_outputs, x.shape[-1]))
    keys = jax.random.split(key, x.shape[-1])
    losses, grads = vgloss(dynamic, static, x, y, z, keys)
    updates, opt_state = optimizer.update(grads, opt_state, dynamic)
    dynamic = optax.apply_updates(dynamic, updates)
    params = ParameterTree.merge(static, dynamic)
    # params = vmap(stack.post_process, in_axes=(-1,), out_axes=-1)(params)
    res = {
        'params': params,
        'losses': losses,
        'grad': grads,
        'opt': opt_state,
    }
    return res


keep_in_history = training_config.get('keep_in_history', ['losses'])



def scannable_step(carry, i_x_y_k):
    params, opt_state = carry
    i, x, y, k = i_x_y_k
    updt = training_step(params, opt_state, x, y, k)
    params, opt_state = updt['params'], updt['opt']
    history = {k: updt[k] for k in keep_in_history}
    return (params, opt_state), history


def epoch_step(start_params, start_opt_state, epoch_key, xbs, ybs):
    pscan = ut.progress_scan(steps_per_epoch, message='Training model')
    batch_keys = jax.random.split(epoch_key, steps_per_epoch)
    sstep = pscan(scannable_step)
    (final_params, final_opt_state), epoch_history = jax.lax.scan(
        sstep,
        (start_params, start_opt_state),
        (jnp.arange(steps_per_epoch), xbs, ybs, batch_keys),
    )
    return final_params, final_opt_state, epoch_history


def no_scan_epoch_step(start_params, start_opt_state, epoch_key, xbs, ybs):
    batch_keys = jax.random.split(epoch_key, steps_per_epoch)
    epoch_history = {}
    params, opt_state = start_params, start_opt_state
    for i, (x, y, k) in enumerate(zip(xbs, ybs, batch_keys)):
        updt = jax.jit(training_step)(params, opt_state, x, y, k)
        params, opt_state = updt['params'], updt['opt']
        history = {k: updt[k] for k in keep_in_history}
        for k, v in history.items():
            epoch_history.setdefault(k, []).append(v)
    return params, opt_state, epoch_history

with ut.timer('Lowering the epoch_step function before compilation'):
    xb = ut.get_looped_slice(xbatches, 0 * steps_per_epoch, steps_per_epoch)
    yb = ut.get_looped_slice(ybatches, 0 * steps_per_epoch, steps_per_epoch)
    lowered = jax.jit(epoch_step).lower(full_params, opt_state, key, xb, yb)
with ut.timer('Compiling the epoch_step function'):
    compiled_epoch_step = lowered.compile()

# compiled_epoch_step = jax.jit(epoch_step)
# compiled_epoch_step = no_scan_epoch_step

##

def clog(epoch, training_config, epoch_history=None, **_):
    if epoch_history is not None and len(epoch_history['losses']) > 0:
        losses = np.array(epoch_history['losses'])
        avg = np.nanmean(losses)
        std = np.nanstd(losses)
        lmin, lmax = np.nanmin(losses), np.nanmax(losses)
        fmt = lambda x: f'{x:.2e}' if x < 1e-4 or x > 1e4 else f'{x:.4f}'
        ut.logger.info(
            f"""[{epoch}/{training_config["epochs"]}] \
        loss: {fmt(avg)} ± {fmt(std)} [min {fmt(lmin)}, max {fmt(lmax)}] in \
        {epoch_history["epoch_time"]:.2f}s"""
        )

# --- main training loop
params = deepcopy(full_params)

if loggers is None:
    loggers = [(1, clog)]

for _, l in loggers:
    l(epoch=0, training_config=training_config)

ut.logger.info(f'Begin training for {training_config["epochs"]} epochs')

all_losses = []

for i, epoch_key in enumerate(jax.random.split(key, training_config['epochs']), 1):

    t0 = time.time()
    xb = ut.get_looped_slice(xbatches, i * steps_per_epoch, (i + 1) * steps_per_epoch)
    yb = ut.get_looped_slice(ybatches, i * steps_per_epoch, (i + 1) * steps_per_epoch)
    params, opt_state, epoch_history = compiled_epoch_step(params, opt_state, epoch_key, xb, yb)
    epoch_history['epoch_time'] = time.time() - t0
    epoch_history['latest_params'] = params
    all_losses.append(epoch_history['losses'])

    for t, l in loggers:
        if t is not None:
            if (t == 0 or (i % t == 0 and t > 0)) or i == training_config['epochs']:
                l(
                    epoch=i,
                    training_config=training_config,
                    epoch_history=epoch_history,
                    nbatches=steps_per_epoch,
                )

##
from labellines import labelLine, labelLines
loss = np.concatenate(all_losses)
loss.shape # (epochs * steps_per_epoch, nruns)
smooth_window = 1000
smoothed_losses = np.array([np.convolve(l, np.ones(smooth_window) / smooth_window, mode='valid') for l in loss.T])
fig, ax = du.mkfig(1, 1, (15, 10))
for i,l in enumerate(smoothed_losses):
    ax.plot(l, label=f'run {i}')
# ax.set_yscale('log')
ax.set_xlabel('epoch')
ax.set_ylabel('loss')
# wirte the legend above each line
labelLines(ax.get_lines(), zorder=2.5)
# plt.show()
# save
fig.savefig('losses.png', dpi=300)

##

res = 100
xlims = (0, 1)

def plot_eval(params, res=100, xlims=(0, 1)):
    X = np.meshgrid(np.linspace(*xlims, res), np.linspace(*xlims, res))
    X = np.stack(X, axis=-1).reshape(-1, 2)
    Z = jnp.ones((res * res, stack.total_nb_of_outputs)) * 0.5
    Y = jax.jit(evaluate_at)(params, X, Z, key)
    fig, ax = du.mkfig(1, 1)
    im = ax.imshow(Y.reshape(res, res), extent=[*xlims, *xlims], cmap='YlGnBu', origin='lower')
    ax.contour(
        Y.reshape(res, res), 2, extent=[*xlims, *xlims], colors='k', origin='lower', alpha=0.25
    )
    # colorbar
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.1)
    fig.colorbar(im, cax=cax, orientation='vertical')
    ax.set_xlabel('$X_1$')
    ax.set_ylabel('$X_2$')
    return fig, ax



for i in range(len(BP)):
    p = jtu.tree_map(lambda x: x[...,i], params)
    f, a = plot_eval(p, res, xlims)
    f.savefig(f'eval_{i}.png', dpi=300)

    # x, y = BP[i]
    # fig, ax = plt.subplots()
    # ax.scatter(x[:, 0], x[:, 1], c=y, s=2, cmap='inferno', alpha=0.75, vmin=0, vmax=0.6)
    # ax.set_xlim(vlims_log)
    # ax.set_ylim(vlims_log)
    # plt.show()


