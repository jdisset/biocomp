# biocomp

` ([x] => done; [.] => doing; [ ] => todo) `

- [x] Write specs for recipe, xp and data files

- [.] Complete the compute graph construction from new recipe file
- [x] Implement TU representation with slots
    - [x] distinguish btwnm parts and params slots
    - [x] implement a constraint system for slots
    - [x] implement resolve_slot
- [x] Generate CDG from TUdf. ! Unresolved slots are not allowed to merge !
- [x] Prototype switch from tree-based to dependency graph-based compute
- [x] Parse recipe
    - [x] Construct a TUdf from fixed recipe
    - [x] Detect free/fixed parameters
- [ ] Add Aggregations.
    - [x] As graph
    - [ ] As compute
- [ ] Add Sources. Basically a no-op splitter?
    - [x] As graph
    - [ ] As compute
- [ ] Add noise distribution to all compute nodes?

- [.] generate_model from the compute graph
- [.] transparent get_quantized()
	- [ ] each compute node should receive a partialed get_quantized method
    - [ ] get_possible_values() from input/output dual CDG (grab them from tuids)
- [.] transparent get_params()
    - [ ] each compute node should receive a partialed get_params method
    - [ ] later: add a condition_on param, that will activate a transform layer on the params. 
           e.g get_params('tc_rate', nid, condition_on='input')
- [ ] Handle inputs

- [ ] Refactor:
- [ ] Rename things consistently. XP -> Recipe (or network?), etc.
- [ ] Move all the compg and cdg creation to the Recipe class, including the generate_model method.
- [ ] Make a separate train module, with a train function that takes a recipe and a dataset as input.

- [ ] Train
- [ ] Add a way to specify params that are fixed vs trainable before traning,
       and aggregate them in a transparent dictionnary that will be passed to the compute graph
       Probably should just split into 2 dictionnaries given to the train method (1st is differentiated against, 2nd is fixed).
       Then do a merge of the 2 before passing them to the CG. Q: will Jax be ok to compile that?
- [ ] Invertible path addition to the compute graph:
    - [ ] make an inverse version of the compute node dictionnary
    - [ ] ensure that each numeric node is tied to an invertible path.
    - [ ] add the inverse path to the compute graph (fluo -> invpath -> numeric -> fwdpath -> fluo)
- [ ] Parse data file (start with Georg's?) and load into dataframe
- [ ] write training loop. Loss = L2 (fluo_out_from_full_gaph, fluo_out_measured)

- [ ] Write tests, especially to test compute graph consistency, especially cdf <-> compg

