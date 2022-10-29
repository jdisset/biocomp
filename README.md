# biocomp

# TODO 

### October 

- [x] Complete the new version of the compute graph construction from new recipe file
    - [x] Implement TU representation with slots
		- [x] distinguish btwnm parts and params slots
		- [x] implement a constraint system for slots
		- [x] implement resolve_slot
	- [x] Generate CDG from TUdf. ! Unresolved slots are not allowed to merge !
	- [x] Prototype switch from tree-based to dependency graph-based compute
	- [x] Parse recipe
		- [x] Construct a TUdf from fixed recipe
		- [x] Detect free/fixed parameters
	- [x] Add Aggregations.
		- [x] As graph
		- [x] As compute
	- [x] Add Sources. Basically a no-op splitter?
		- [x] As graph
		- [x] As compute
	- [ ] FOR LATER: add noise distribution layer to all compute nodes?

- [x] generate_model from the compute graph
	- [x] transparent get_quantized()
		- [x] each compute node should receive a partialed get_quantized method
		- [x] get_possible_values() from input/output dual CDG (grab them from tuids)
	- [x] transparent get_params()
		- [x] each compute node should receive a partialed get_params method
		- [ ] FOR LATER: add a condition_on param, that will activate a transform layer on the params. 
			   e.g get_params('tc_rate', nid, condition_on='input')
	- [x] Handle inputs
	- [x] Invertible path addition to the compute graph:
		- [x] make an inverse version of the compute node dictionnary
		- [x] ensure that each numeric node is tied to an invertible path.
		- [x] add the inverse path to the compute graph (fluo -> invpath -> numeric -> fwdpath -> fluo)

- [ ] Refactor:
	- [x] Rename things consistently. XP -> Recipe (or network?), etc.
	- [x] Move all the compg and cdg creation to the Recipe class, including the generate_model method.
	- [ ] Make a separate train module, with a train function that takes a recipe and a dataset as input.

- [x] Data & training:
	- [x] Write specs for recipe, xp and data files
	- [x] Get recipe
	- [x] Get data
	- [x] Add a way to specify params that are fixed vs trainable before traning,
		   and aggregate them in a transparent dictionnary that will be passed to the compute graph
		   Probably should just split into 2 dictionnaries given to the train method (1st is differentiated against, 2nd is fixed).
		   Then do a merge of the 2 before passing them to the CG. Q: will Jax be ok to compile that?
	- [x] Parse data file (start with Georg's?) and load into dataframe
	- [x] write training loop. Loss = L2 (fluo_out_from_full_gaph, fluo_out_measured)

- [ ] Write tests, especially to test compute graph consistency, especially cdf <-> compg


### November

- [ ] Cleaner / shorter train module that uses a more thorough 
config dict (include node remaps and data rebalancing params)

- [ ] Better batches with padding to get a unique across the whole dict

- [ ] Improve accuracy of xp training
	- [ ] Try more complex transcription / translation equations
	- [ ] Try ERN version that takes DNA - RNA instead of RNA - PRT
	- [ ] Try different meta params (norm factor for example)
	- [ ] Try switching ERN to dense NN

- [ ] Write system to store shared parameters, together with network
config (node remaps and list of enabled sequestrons). Need to be able 
to load both.
	- [ ] Cleanly save everything to wandb
	- [ ] Write quick datautils (or train module) function to load 
	from wandb with a run name

- [ ] Test the circuit optimization/exploration mode (with a bandpass 
or maybe just a simple decision boundary).
	_Questions:_ should we force multiple coTX? Because the easiest 
	for me is probably to just have one marker, and determine the 
	ratio of each plasmid. But that might not be the experimentally 
	ideal protocol...
	- [ ] Write set of TUs + constraints to explore N-ERN 
	configurations. Maybe with ERN affinity for now?
	- [ ] Perform sensitivity analysis on each parameter (from all 
	scopes)
	- [ ] Write to output recipe

- [ ] Train on Charles' uOrfs data when it's ready
 
- [ ] Write recombinase based sequestron as a ternary node
 
- [ ] Train on Charles' recombinase data

- [ ] Add noise distribution to all compute nodes?
