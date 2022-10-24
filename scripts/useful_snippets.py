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



