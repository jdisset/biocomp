import biocomp as bc
import biocomp.compute as bcc
import scriptutils as ut
import jax
import jax.numpy as jnp
import random
import json5

random.seed()

lib = ut.load_lib()


recipe_paths = [
    '/Users/jeandisset/Documents/(1_1; 101+104)+(2_1; 102+103).recipe.json5',
    '/Users/jeandisset/Documents/(1_1; 101+116)+(2_1; 102+117)+73.recipe.json5',
]

from tqdm import tqdm

recipes = [json5.load(open(r)) for r in tqdm(recipe_paths)]

recipes[0]

import sqlite3
base_conn = sqlite3.connect(':memory:')
bc.recipe.recipes_to_sql(recipes, base_conn, lib)

n0 = bc.Network(lib, '(1:1; 101+104)+(2/1; 102+103)', base_conn)
n1 = bc.Network(lib, '(1:1; 101+116)+(2/1; 102+117)+73', base_conn)
base_conn.close()


ut.plot_networks([n0, n1], ['n0.pdf', 'n1.pdf'])
