import biocomp.utils as ut
import biocomp.recipe as rc

##

lib = ut.load_lib()
recipe_path = ut.DEFAULT_RECIPE_PATH
RECIPE = "tagBFP.recipe.json5"
network = rc.network_from_recipe(recipe_path[1] / RECIPE, lib)[0]

##
network



