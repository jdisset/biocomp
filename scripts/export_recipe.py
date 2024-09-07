from biocomp import utils as ut
from biocomp import compute as cmp
from biocomp import library
import biocomp as bc
import jax
from pathlib import Path
from omegaconf import OmegaConf
import json
import copy
import pandas as pd
import itertools

# Look at CMAES (Covariant Matrix Augmentation Evolutionary Strategy)
# Look at kickstart.nvim

cat_to_lib_cat = {
    "degron": "3'UTR",
    "rcb_rec_3p": "3'UTR",
    "ERN": "gene",
    "fluo_marker": "gene",
    "inverted_seq": "gene",
    "recombinase_fwd": "gene",
    "recombinase_bwd": "gene",
    "rcb_rec_5p": "5'UTR",
    "ERN_recog_site_5p": "5'UTR",
    "uORF_group": "5'UTR",
    "promoter": "promoter",
    "insulator": "insulator",
    "terminator": "terminator",
    "untranscripted": "insulator",
    "transcripted": "5'UTR",
    "translated": "gene",
    "spacer": "gene",
    "ERN_recog_site_3p": "3'UTR",
}


def resolve_parameters(names: dict[int, str], cdg: pd.DataFrame):
    def propagate_parameters(index: int, cdg: pd.DataFrame, update_dict: dict[str, list[str]]):
        cdg.at[index, "params"].update(update_dict)
        predecessors: list[int] | None = cdg.at[index, "predecessor"]
        if isinstance(predecessors, list):
            for i in predecessors:
                propagate_parameters(i, cdg, update_dict)

    for index, param in names.items():
        update_dict: dict[str, list[str]] = {}
        for param_type, options in cdg.at[index, "params"].items():
            if param in options:
                assert param_type not in update_dict
                update_dict[param_type] = [param]
        propagate_parameters(index, cdg, update_dict)


def resolve_content(index: int, cdg: pd.DataFrame) -> list[str]:
    return list(itertools.chain(cdg.at[index, "content"], *cdg.at[index, "params"].values()))


def order_plasmid(plasmid: list[str], lib: library.PartsLibrary) -> list[str]:
    lib_cat_to_index: dict[str, int] = {cat: ind for ind, cat in enumerate(lib.L1s.columns)}
    new_plasmid: list[list[str]] = [[] for _ in lib_cat_to_index]
    for part in plasmid:
        category = cat_to_lib_cat[lib.parts.at[part, "category"]]
        new_plasmid[lib_cat_to_index[category]].append(part)
    return list(itertools.chain.from_iterable(new_plasmid))


def commit(stack: cmp.ComputeStack):
    params: bc.parameters.ParameterTree = stack.init(key)
    stack.commit(params)
    for network in stack.networks:
        cdg = copy.deepcopy(network.central_dogma_graph)
        names: dict[int, str] = {}
        assert network.compute_graph is not None
        for node in network.compute_graph.itertuples():
            if "resolved_parameter_names" in node.extra:
                resolved_parameter_names: dict[str, str] = node.extra["resolved_parameter_names"]
                assert len(node.extra["resolved_parameter_names"]) == len(node.cdg_input)
                names.update(dict(zip(node.cdg_input, node.extra["resolved_parameter_names"])))
        resolve_parameters(names, cdg)
        aggregations: list[dict[str, list[dict[str, float | str | list[str]]]]] = []
        for node in network.compute_graph.itertuples():
            if node.type == "aggregation":
                sources: list[dict[str, float | str | list[str]]] = []
                assert len(node.output_to) == len(node.extra["ratios"])
                for (plasmid_node, _), ratio in zip(node.output_to, node.extra["ratios"]):
                    for cdg_index in network.compute_graph.loc[plasmid_node]["cdg_output"]:
                        plasmid: list[str] = resolve_content(cdg_index, cdg)
                        sources.append(
                            {"ratio": float(ratio), "plasmid": order_plasmid(plasmid, lib)}
                        )
                aggregations.append({"sources": sources})
        print(json.dumps({"content": aggregations}, indent=3))


config = cmp.ComputeConfig.from_dict(
    OmegaConf.to_container(
        ut.load_config("biocomp/biocomp_default_config/compute_config.yaml"), resolve=True
    )
)

lib: bc.library.PartsLibrary = ut.load_lib()
recipe_path = '/home/gispisquared/Dropbox/Biocomp/Recipes/uORFsum0x.recipe.json5'
recipe_file = Path(recipe_path).expanduser().resolve()
networks: list[bc.Network] = bc.recipe.network_from_recipe(recipe_file, lib, inverse='shortest')

stack: cmp.ComputeStack = cmp.ComputeStack(networks)
stack.build(config)
key: jax.Array = jax.random.PRNGKey(0)
commit(stack)
