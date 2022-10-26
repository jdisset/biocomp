# get lib
def get_lib():
    return ut.getLibFromGoogleSheet()
lib = get_lib()

#  plot a bunch of networks to pdf
import nest_asyncio
nest_asyncio.apply()
networks = list(xp.networks.values())
filenames = [f'../__out/nets/{n.name}.pdf' for n in networks]
ut.plot_networks(networks, filenames)



# plot binstats and heatmap
out_proteins = model.get_output_proteins()
in_proteins = model.get_inverted_input_proteins()
stats, bins = binstats(y, out_proteins, in_proteins, resolution=0.5)
heatmap(
        stats,
        bins,
        figscale=0.6,
        stat_columns=['mean','count'],
        z_protein='eYFP',
        lims={'mean': (1e3, 1e8)},
        title=f'{model.network.name}',
        subtitle=f'{len(y)} data points',
    )

