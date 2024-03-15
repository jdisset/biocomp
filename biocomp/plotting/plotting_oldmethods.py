# ────────────────────────────   archive   ─────────────────────────────
### {{{                      --     density plot 1D     --
def density_plot_1d(
    x,
    sample_at,
    ax,
    color='k',
    label=None,
    ticks=None,
    ticks_labels=None,
    bw_method=None,
    x2=None,
    show_quantiles=[0.005, 0.995],
    **kw,
):
    if bw_method is None:
        bw_method = 0.01
    left_kde = gaussian_kde(x.T, bw_method=bw_method)
    left_densities = left_kde(sample_at.T)
    if x2 is not None:
        right_kde = gaussian_kde(x2.T, bw_method=bw_method)
        right_densities = right_kde(sample_at.T)
    else:
        x2 = x
        right_kde = left_kde
        right_densities = left_densities

    left_densities = (left_densities / left_densities.max()) * 0.4
    right_densities = (right_densities / right_densities.max()) * 0.4

    ax.plot(-left_densities, sample_at, color='k', alpha=1, lw=0.5)
    ax.plot(right_densities, sample_at, color='k', alpha=1, lw=0.5)

    if show_quantiles is not None:
        maxleft = sample_at[left_densities.argmax()]
        q1 = np.quantile(x, show_quantiles[0])
        q9 = np.quantile(x, show_quantiles[-1])
        ax.plot([-0.5, 0], [q1, q1], color=color, lw=1)
        ax.plot([-0.5, 0], [q9, q9], color=color, lw=1)
        # ax.plot([-0.5, 0], [maxleft, maxleft], color='k', lw=1)
        ax.fill_betweenx([q1, q9], -0.5, 0, color=color, alpha=0.1, lw=0)
        maxright = sample_at[right_densities.argmax()]
        q1 = np.quantile(x2, show_quantiles[0])
        q9 = np.quantile(x2, show_quantiles[-1])
        ax.plot([0, 0.5], [q1, q1], color=color, lw=1)
        ax.plot([0, 0.5], [q9, q9], color=color, lw=1)
        # ax.plot([0, 0.5], [maxright, maxright], color='k', lw=1)
        ax.fill_betweenx([q1, q9], 0, 0.5, color=color, alpha=0.1, lw=0)

    ax.fill_betweenx(sample_at, -left_densities, 0, color=color, alpha=1, lw=0)
    ax.fill_betweenx(sample_at, 0, right_densities, color=color, alpha=1, lw=0)
    ax.axvline(0, color='k', alpha=0.5, lw=0.5, dashes=(10, 10), dash_capstyle='round')
    ax.set_aspect("equal")
    ax.set_xlim(-0.5, 0.5)
    remove_axis_and_spines(ax)
    if label is not None:
        ax.set_xlabel(label, rotation=0, labelpad=20, fontsize=10)
    if ticks is not None:
        for t in ticks:
            ax.axhline(
                t,
                xmin=-0.2,
                xmax=1,
                c='#777777',
                linewidth=0.2,
                zorder=0,
                clip_on=False,
                alpha=1,
                dashes=(10, 20),
                dash_capstyle='round',
            )
        if ticks_labels is not None:
            ax.set_yticks(ticks)
            ax.set_yticklabels(ticks_labels)
            ax.tick_params(axis='y', which='both', length=0, pad=30)
            for tick in ax.yaxis.get_major_ticks():
                tick.label.set_fontsize(8)
                tick.label.set_color('grey')


##────────────────────────────────────────────────────────────────────────────}}}
### {{{        --     [OLD] 2D    --
def prepare_smooth_2d(
    x,
    y,
    network,
    input_names,
    input_order,
    output_pos,
    res=200,
    xlims=(0, 1),
    xslice=None,
    density_plot=False,
    density_as_alpha=False,
    density_threshold=10,
    use_y_as_x=False,  # if True, use the output of the independent variables as coordinates
    **kw,
):
    xmin, xmax = xlims

    if use_y_as_x:
        output_names = network.get_output_proteins()
        xind = [output_names.index(i) for i in input_names]
        x = y[:, xind]
    else:
        x = x[:, input_order]

    y = y[:, output_pos]

    xy = make_xy_grid(xmin, xmax, xres=res)
    if x.shape[1] > 2:
        assert xslice.shape == (x.shape[1] - 2,)
        xquery = np.concatenate([xy, np.tile(xslice, (xy.shape[0], 1))], axis=1)
    else:
        xquery = xy

    tree = cKDTree(x)
    output_values, density = knn_avg(xquery, y, tree=tree, **kw)
    assert output_values.shape == (xy.shape[0],)
    assert density.shape == (xy.shape[0],)
    opacities = (
        np.ones_like(density)
        if not density_as_alpha
        else np.minimum(density / density_threshold, 1.0)
    )
    opacities = np.where(np.isnan(output_values), 1, opacities)
    if density_plot:
        output_values = density

    return xy, output_values, opacities


def smooth_2d_old(
    x,
    y,
    network,
    rescaler,
    ax,
    res=200,
    xlims=(0, 1),
    xslice=None,  # should be called zslice, really...
    title=None,
    text_x=0.5,
    text_y=0.9,
    axtransform=None,
    show_slice_title=True,
    **kw,
):

    protein_order, protein_names = get_reordered_protein_names(network, **kw)
    input_order, output_pos = protein_order[:-1], protein_order[-1]
    input_names, output_name = protein_names[:-1], protein_names[-1]
    # remove input_order from kw
    kw.pop('input_order', None)

    xy, output_values, opacities = prepare_smooth_2d(
        x, y, network, input_names, input_order, output_pos, res, xlims, xslice, **kw
    )

    hm = heatmap(
        ax, xy, output_values, rescaler, opacities=opacities, axtransform=axtransform, **kw
    )

    ax.set_xlabel(input_names[0])
    ax.set_ylabel(input_names[1])

    full_transform = ax.transData if axtransform is None else ax.transData + axtransform

    if x.shape[1] > 2 and show_slice_title:
        ax.text(
            text_x,
            text_y,
            f'{input_names[2]} $ \\approx $ {format_powers(rescaler.inv(xslice[0]), n_decimals=0)}',
            fontsize=5,
            transform=full_transform,
            ha='center',
            va='bottom',
        )

    # spines only on bottom and left
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ttle = None
    if title is True:
        ttle = f'{network.name}\n{output_name} smoothed mean'
    elif title is not None:
        ttle = title
    if ttle is not None:
        ax.set_title(ttle)

    return hm


##────────────────────────────────────────────────────────────────────────────}}}
### {{{                       --     [OLD]    density histogram     --


def histogram_old(x, y, network, rescaler, ax, **kw):
    ninputs = network.get_nb_inputs()
    if ninputs == 1:
        histogram_plot_old(x, y, network, rescaler, ax, **kw)
    else:
        raise NotImplementedError(f'Cannot plot {ninputs} inputs')


def histogram_plot_old(
    X,
    Y,
    network,
    rescaler,
    ax,
    nbins=256,
    xlims=(0, 1),
    ylims=(0, 1),
    vlims=(0.001, None),
    cmap=DEFAULT_CMAP_NAME,
    noise_smooth=0,
    log_density=True,
    **kw,
):
    assert X.shape[1] == 1

    if isinstance(nbins, int):
        nbins = [nbins, nbins]

    protein_order, protein_names = get_reordered_protein_names(network, **kw)
    input_order, output_pos = protein_order[:-1], protein_order[-1]
    input_names, output_name = protein_names[:-1], protein_names[-1]

    Y = Y[:, output_pos]
    X = X[:, 0]

    xres = np.abs(np.subtract(*xlims)) / nbins[0]
    yres = np.abs(np.subtract(*ylims)) / nbins[1]

    X = X + np.random.normal(size=X.shape) * noise_smooth * xres
    Y = Y + np.random.normal(size=Y.shape) * noise_smooth * yres

    h, xedges, yedges = np.histogram2d(
        X,
        Y,
        bins=nbins,
        density=False,
        range=[xlims, ylims],
    )

    if log_density:
        h = np.log10(h + 1)

    xlims_true_scale, ylims_true_scale = setup_transformed_axis(
        ax,
        xaxis_lims=xlims,
        yaxis_lims=ylims,
        rescaler=rescaler,
        margins=0.0,
        **kw,
    )

    h = h.T  # matplotlib wants it transposed
    ax.imshow(
        h,
        extent=[*xlims, *ylims],
        origin='lower',
        aspect='auto',
        cmap=cmap,
        vmin=vlims[0],
        vmax=vlims[1],
    )

    ax.set_xlabel(input_names[0])
    ax.set_ylabel(output_name)

    # show grid, including minor grid
    ax.grid(color='k', alpha=0.25, linestyle='-', linewidth=0.2, which='major')
    ax.grid(color='k', alpha=0.1, linestyle='-', linewidth=0.1, which='minor')


##────────────────────────────────────────────────────────────────────────────}}}
### {{{                            --     3D     --
def smooth_3d(
    x, y, network, rescaler, slices=np.linspace(0, 0.65, 4), axes=None, top_ax=None, **kw
):
    assert axes is not None
    if len(axes) != len(slices):
        raise ValueError(
            f'axes and slices must have the same length, got {len(axes)} and {len(slices)}'
        )

    porder, pnames = get_reordered_protein_names(network, **kw)

    for i, s in enumerate(slices):

        def get_cbar_ticks(vmin, vmax):
            (
                in_order,
                in_names,
                out_pos,
                out_name,
                vticks,
                vtlabels,
                secondticks,
            ) = network_ticks_and_labels(network, rescaler, xmin=vmin, xmax=vmax, **kw)
            return vticks, vtlabels

        kw.pop('ax', None)
        smooth_2d(
            x,
            y,
            network,
            rescaler,
            ax=axes[i],
            xslice=np.array([slices[i]]),
            get_cbar_ticks=get_cbar_ticks,
            **kw,
        )

    if top_ax is not None:
        top_ax.set_xlabel(pnames[-2])
        default_style(top_ax)
        top_ax.spines['left'].set_visible(False)

    # resize all axes  so that they are square and fit in the original ax
    for i, a in enumerate(axes):
        if len(a.get_images()) > 0:
            if i > 0:
                a.set_ylabel('')
            if i < len(axes) - 1:
                a.get_images()[0].colorbar.remove()
            else:
                # write the label on the right of the colorbar
                cbarax = a.get_images()[0].colorbar.ax
                cbarax.yaxis.set_label_position('right')
                cbarax.set_ylabel(pnames[-1], fontsize=8)

        a.set_title('')


##────────────────────────────────────────────────────────────────────────────}}}
## {{{                      --     smooth dispatch     --
def smooth_old(x, y, network, rescaler, **kw):
    ninputs = network.get_nb_inputs()
    if ninputs == 1:
        smooth_1d(x, y, network, rescaler, **kw)
    elif ninputs == 2:
        smooth_2d_old(x, y, network, rescaler, **kw)
    elif ninputs == 3:
        smooth_3d(x, y, network, rescaler, **kw)
    else:
        raise NotImplementedError(f'Cannot plot {ninputs} inputs')
##────────────────────────────────────────────────────────────────────────────}}}


# ---- summary model plots + misc
### {{{                --     summary model plot functions     --
def network_plot(
    dman: DataManager,
    network_id: int,
    *args,
    kde=None,
    density_quantile_threshold=0.05,
    use_xy=None,
    method='smooth',
    **kw,
):
    network = dman.get_networks()[network_id]
    if use_xy is None:
        x, y = dman.get_X()[network_id], dman.get_Y()[network_id]
    else:
        x, y = use_xy

    if kde is not False:
        if kde is None:
            kde = dman.get_kdes()[network_id]

    rescaler = DataRescaler.from_data_manager(dman)

    return direct_network_plot_old(
        network,
        x,
        y,
        rescaler,
        *args,
        kde=kde,
        density_quantile_threshold=density_quantile_threshold,
        method=method,
        **kw,
    )


def direct_network_plot_old(
    network,
    x,
    y,
    rescaler,
    *args,
    kde=None,
    density_quantile_threshold=0.05,
    method='smooth',
    **kw,
):
    if kde is not False and kde is not None:
        rng = jax.random.PRNGKey(0)
        subsample = du.optimal_density_subsample(
            x, kde, rng, quantile_threshold=density_quantile_threshold
        )
        x, y = x[subsample], y[subsample]
    if method == 'smooth':
        return smooth_old(x, y, network, rescaler, *args, **kw)
    elif method == 'scatter':
        return scatter(x, y, network, rescaler, *args, **kw)
    elif method == 'histogram':
        return histogram(x, y, network, rescaler, *args, **kw)
    elif method == 'smooth_line_slices':
        return smooth_line_slices(x, y, network, rescaler, *args, **kw)


def eval_network_plot(
    params,
    dman,
    id,
    ax,
    npoints_eval=20000,
    quantile_range=[0.2, 0.8],
    key=jax.random.PRNGKey(0),
    xrange_eval=None,
    **kw,
):
    k_i, k_q = jax.random.split(key)
    if xrange_eval is None:
        xrange_eval = np.array([[0, 0], [1, 1]])

    network = dman.get_networks()[id]
    jm = jit(dman.get_individual_compute_stack(id).apply)

    x = jax.random.uniform(
        k_i, (npoints_eval, network.get_nb_inputs()), minval=xrange_eval[0], maxval=xrange_eval[1]
    )
    quantiles = jax.random.uniform(
        k_q, (npoints_eval, network.n_outputs), minval=quantile_range[0], maxval=quantile_range[1]
    )
    keys = jax.random.split(key, npoints_eval)
    y = vmap(jm, in_axes=(None, 0, 0, 0))(params, x, quantiles, keys)

    xmin, xmax = np.min(x, axis=0)[0], np.max(x, axis=0)[0]

    smooth_old(x, y, network, dman.rescale, ax, xmin=xmin, xmax=xmax, **kw)


def get_stack(dman, net_id, params):
    stack, pf = dman.get_individual_compute_stack(net_id)
    p = pf(params)
    return stack, p


def eval_network_on_grid(
    params,
    network,
    stack,
    ax,
    rescale,
    key=jax.random.PRNGKey(0),
    xrange_eval=(0, 1),
    n_repeats=10,
    quantile_range=(0.2, 0.8),
    res=100,
    **kw,
):
    jm = jit(stack.apply)

    (
        input_order,
        input_names,
        output_pos,
        output_name,
        ticks,
        tlabels,
        secondticks,
    ) = network_ticks_and_labels(network, rescale, xmax=xrange_eval[1], **kw)

    k_i, k_q = jax.random.split(key)
    if xrange_eval is None:
        xrange_eval = np.array([0, 1])

    xx = np.linspace(xrange_eval[0], xrange_eval[1], res)
    x = np.array(np.meshgrid(xx, xx)).T.reshape(-1, 2)

    def compute(k):
        quantiles = jax.random.uniform(
            k,
            (len(x), network.get_nb_outputs()),
            minval=quantile_range[0],
            maxval=quantile_range[1],
        )
        keys = jax.random.split(k, len(x))
        y, _ = vmap(jm, in_axes=(None, 0, 0, 0))(params, x, quantiles, keys)
        return y

    keys = jax.random.split(key, n_repeats)
    all_y = vmap(compute)(keys)
    y_mean = np.mean(all_y, axis=0)

    z = y_mean[:, output_pos]
    z = z.reshape(res, res)

    # heatmap(ax, z, ticks=ticks, ticklabels=tlabels, **kw)
    ax.set_xlabel(input_names[0])
    ax.set_ylabel(input_names[1])
    remove_spines(ax)


def eval_model_grid(
    params,
    dman,
    id,
    ax,
    **kw,
):
    network = dman.get_networks()[id]
    stack, p = get_stack(dman, id, params)
    return eval_network_on_grid(params, network, stack, ax, rescale=dman.rescale, **kw)


def model_at_x(params, dman: DataManager, id, key=jax.random.PRNGKey(0), quantile=None, **_):
    stack, p = get_stack(dman, id, params)

    x, y = dman.get_X()[id], dman.get_Y()[id]
    keys = jax.random.split(key, x.shape[0])

    if quantile is not None:
        Q = jnp.ones(y.shape) * quantile
    else:
        Q = jax.random.uniform(key, y.shape)

    yhat, _ = jit(vmap(stack.apply, in_axes=(None, 0, 0, 0)))(p, x, Q, keys)

    return x, y, yhat


def plot_model_at_x(params, dman, id, ax, **kw):
    x, _, yhat = model_at_x(params, dman, id, **kw)
    net = dman.get_networks()[id]
    smooth_old(x, yhat, net, dman.rescale, ax, **kw)


def plot_model_diff(params, dman, id, ax, **kw):
    x, y, yhat = model_at_x(params, dman, id, **kw)
    net = dman.get_networks()[id]
    err = np.abs(y - yhat)
    smooth_old(x, err, net, dman.rescale, ax, **kw)


def report(params, dman, id, suptitle='', use_x_y_yhat=None, **kw):
    if use_x_y_yhat is not None:
        x, y, yhat = use_x_y_yhat
        assert len(x) == len(y), 'x and y must have the same length'
        assert y.shape == yhat.shape, 'y and yhat must have the same shape'
        ndim = x.shape[1]
        if ndim <= 2:
            fig, ax = mkfig(1, 2, size=(4, 4))
            network_plot(dman, id, ax[0], use_xy=(x, y), kde=False, **kw)
            network_plot(dman, id, ax[1], use_xy=(x, yhat), kde=False, **kw)
            ax[0].set_title(f'Original data (mean)')
            ax[1].set_title(f'Predicted (mean)')
        elif ndim == 3:
            fig, axes = mkfig(2, 4, size=(4, 4))
            contours = np.linspace(0, 0.8, 5)
            top_row_axes = axes[0, :]
            bottom_row_axes = axes[1, :]
            slices = (np.linspace(0.1, 0.8, 4),)
            network_plot(
                dman,
                id,
                ax=None,
                axes=top_row_axes,
                contours=contours,
                slices=np.linspace(0.1, 0.8, 4),
                use_xy=(x, y),
                **kw,
            )
            network_plot(
                dman,
                id,
                ax=None,
                axes=bottom_row_axes,
                contours=contours,
                slices=np.linspace(0.1, 0.8, 4),
                use_xy=(x, yhat),
                **kw,
            )
            for ax in axes.flatten():
                ax.set_title('')
            axes[0, 0].set_title(f'Original data (mean)')
            axes[1, 0].set_title(f'Predicted (mean)')
        else:
            raise ValueError(f'ndim={ndim} not supported')
    else:
        fig, ax = mkfig(1, 2, size=(4, 4))
        network_plot(dman, id, ax[0], **kw)
        plot_model_at_x(params, dman, id, ax[1], **kw)
        ax[0].set_title(f'Original data (mean)')
        ax[1].set_title(f'Predicted (mean)')

    network = dman.get_networks()[id]
    fig.suptitle(f'{suptitle} {network.name}')
    fig.tight_layout()
    return fig


##────────────────────────────────────────────────────────────────────────────}}}
### {{{                  --     Fluo distribution plots     --


def fluo_scatter(
    rawx,
    pnames,
    xmin=0,
    xmax=None,
    title=None,
    types=None,
    fname=None,
    logscale=True,
    alpha=0.1,
    maxn=50000,
    s=2,
    **_,
):
    fig, axes = plt.subplots(1, len(pnames), figsize=(1.25 * len(pnames), 10), sharey=True)

    if len(pnames) == 1:
        axes = [axes]

    if types is None:
        types = [''] * len(pnames)

    if xmin is None:
        xmin = rawx.min()
    if xmax is None:
        xmax = rawx.max()

    X = rawx.copy()
    if len(X) > maxn:
        X = X[np.random.choice(len(X), maxn, replace=False)]

    tr = lambda x: x
    itr = tr
    for xid, ax in enumerate(axes):
        color = get_bio_color(pnames[xid])
        xcoords = np.random.normal(0, 0.1, (X.shape[0],))
        if logscale:
            tr, itr, _, ytr = setup_symlog_axis(ax, None, yaxis_lims=[xmin, xmax])
        else:
            ax.set_ylim(xmin, xmax)
        ax.scatter(xcoords, tr(X[:, xid]), color=color, alpha=alpha, s=s, zorder=10, lw=0)

        remove_spines(ax)
        ax.set_xlim(-0.5, 0.5)
        ax.set_xlabel(f'{pnames[xid]} {types[xid]}', rotation=0, labelpad=20, fontsize=10)
        ax.set_xticks([])

    if title is not None:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    if fname is not None:
        fig.savefig(fname)

    return fig, axes


def fluo_densities(
    rawx, pnames, xmin=None, xmax=None, res=1000, title=None, types=None, logscale=False, **kw
):
    fig, axes = plt.subplots(1, len(pnames), figsize=(1.5 * len(pnames), 10))

    # def loglog(x):
    # return np.where(x > 1, np.log10(x), np.where(x < -1, -np.log10(-x), 0))

    # def inv_loglog(x):
    # return np.where(x > 0, 10**x, np.where(x < 0, -(10**-x), 0))

    if logscale:
        X = loglog(rawx)
    else:
        X = rawx
    xmin = xmin if xmin is not None else np.floor(X.min())
    xmax = xmax if xmax is not None else np.ceil(X.max())

    ticks = np.arange(xmin, xmax + 1, 1)
    sample_at = np.linspace(xmin, xmax, res)

    if logscale:
        ylabels = [scformat.format("{:m}", x) for x in inv_loglog(ticks)]
    else:
        ylabels = [scformat.format("{:m}", x) for x in ticks]

    if types is None:
        types = [''] * len(pnames)
    for xid, ax in enumerate(axes):
        color = get_bio_color(pnames[xid], default='#AAAAAA')
        tlabels = ylabels if xid == 0 else None
        density_plot_1d(
            X[:, xid],
            sample_at,
            ax,
            color=color,
            label=f'{pnames[xid]} {types[xid]}',
            ticks=ticks,
            ticks_labels=tlabels,
            **kw,
        )
        ax.set_ylim(xmin, xmax)
    if title is not None:
        fig.suptitle(
            title,
            fontsize=10,
            y=0.95,
            x=0.45,
        )
    fig.tight_layout()
    return fig, axes


def model_fluo_distributions(dman, model_id, method='scatter', **kwargs):
    model = dman.get_models()[model_id]
    rawx = dman.get_raw_X()[model_id]
    rawy = dman.get_raw_Y()[model_id]
    input_names = model.get_inverted_input_proteins()
    reordered_input = sorted(input_names)
    output_names = model.get_output_proteins()
    output = list(set(output_names) - set(input_names))
    output_pos = output_names.index(output[0])
    if reordered_input != input_names:
        rawx = rawx[:, [input_names.index(i) for i in reordered_input]]
    rawx = np.hstack([rawx, rawy[:, output_pos][:, None]])
    pnames = reordered_input + output
    types = ['[in]'] * len(reordered_input) + ['[out]']
    if method == 'scatter':
        fluo_scatter(rawx, pnames, types=types, **kwargs)
    elif method == 'kde':
        fluo_densities(rawx, pnames, types=types, **kwargs)


##────────────────────────────────────────────────────────────────────────────}}}
### {{{                   --     node functions plots     --
def plot_node(
    node_name,
    shared_parameters,
    compute_config,
    ax,
    median_evals_resolution=200,
    n_random_evals=10000,
    xlims=(0, 1),
    color='k',
    quantized_param_id=0,
):
    tl = compute_config.get_impl(node_name)

    L = tl(input_shapes=[(1,)], n_outputs=1, stack=None, layer_id=0)

    class FakeNode(cmp.VirtualNode):
        def get_compute_node(self, _):
            return None

        def get_inverse_node(self, _):
            return None

        def get_layer_and_local_id(self, _):
            return 0, 0

    key = jax.random.PRNGKey(0)

    p = pm.ParameterTree()
    L.prepare(p, [FakeNode()], key)
    p.tag('local', 'local')
    local, _ = p.filter_by_tag('local')

    qname = None
    qnames = []
    if node_name in ('translation', 'transcription', 'inv_transcription', 'inv_translation'):
        qmaskleaf = None
        for l, v in local.data.iter_leaves():
            if str(l).endswith('quantization_mask'):
                qmaskleaf = l
                break
        qnames = compute_config.config['functions'][node_name]['parameters']['quantization_names']
        base_mask = np.zeros((len(qnames),), dtype=np.bool).reshape(1, 1, -1)
        base_mask[:, :, quantized_param_id] = True
        local[qmaskleaf] = base_mask
        qname = qnames[quantized_param_id]

    pmerged = pm.ParameterTree.merge(shared_parameters, local)

    @jax.jit
    def vapply(xvals, qs, params):
        f = lambda x, q: L.apply(x, quantiles=q, node_id=0, params=params, key=key)
        return jax.vmap(f)(xvals, qs)

    x = np.linspace(*xlims, median_evals_resolution).reshape(-1, 1)
    medianq = np.ones_like(x) * 0.5

    ymedian = vapply(x, medianq, pmerged).flatten()

    n_random_evals = 20000
    randomx = np.random.uniform(0, 1, n_random_evals).reshape(-1, 1)
    randomq = np.random.uniform(0, 1, n_random_evals).reshape(-1, 1)
    yrandom = vapply(randomx, randomq, pmerged).flatten()

    # from qid

    ax.scatter(randomx, yrandom, s=2, c=color, alpha=0.05, linewidth=0)
    ax.plot(x, ymedian, label=qname if qname is not None else '', c=color, ls='--', lw=2)


##────────────────────────────────────────────────────────────────────────────}}}
### {{{                    --     High level helpers     --

BASE_DEFAULT_CONFIG = {
    'xlims': (-0.027, 0.8),
    'ylims': (-0.027, 0.8),
    'log_density': True,
    'size': (4, 4),
    'skip_ticklabel_range': (0.0, 101),
}

DEFAULT_1D_CONFIG = {
    'method': 'histogram',
}

DEFAULT_2D_CONFIG = {
    'method': 'smooth',
}

DEFAULT_3D_CONFIG = {
    'xlims': (-0.027, 0.85),
    'ylims': (-0.027, 0.85),
    'vlims': (-0.027, 0.85),
    'method': 'smooth',
    'slices': (0.1, 0.3, 0.5),
    'radius': 0.11,
    'knn': 500,
    'min_points': 20,
}


##────────────────────────────────────────────────────────────────────────────}}}
