# Overview of the Biocompiler

Neuromorphic circuits in cells represent a categorical jump from the currently
dominant boolean logic paradigm. The modularity of Sequestrons, the wide analog
dynamic range they can output in and take inputs from, and the vast array of
available biological parameters that can shape their behavior put neuromorphic
circuits one step closer to universal biological function approximation.
However, just like with their in-silico neural counterparts, this composable
complexity comes at the cost of a large, rapidly increasing combinatorial
space.

The parallel with artificial neural networks is useful to reason about
the general properties and possibilities of neuromorphic circuits, but a few
key differences add to the complexity of designing and predicting the behavior
of neuromorphic circuits:

- The sum of inputs to a Sequestron is a complex function of input values and
  weights (e.g., number of uORFs, promoter strength), rather than a simple linear
  combination.

- Each Sequestron implementation (e.g., CasE, Csy4, PgU) has a distinct
  non-linear response to its inputs.

- The quantities being handled are not abstract numbers, but biological
  quantities with categorical and non-linear properties. The network also needs
  to handle a mix of continuous variables (species concentrations) and discrete
  ones (circuit components)

Just as backpropagation and modern machine learning training techniques are
central to the exploration of vast parameter spaces in deep learning, we argue
that an algorithmic in silico approach is essential for designing and
predicting the behavior of neuromorphic circuits. This is crucial for
navigating their large design space and unlocking their full potential as a new
cellular programming paradigm.

We developed a machine learning framework with two main objectives:

- Predict the behavior of cells based on the DNA they are transfected with and
  the absolute quantities of each transcription unit.

- Design and tune novel circuits that achieve a desired function.

This framework is capable of training with flow cytometry data from a limited
set of circuit architectures.

At its core is a novel machine learning architecture we call a biomorphic neural
network (BNN). A BNN is a composition of neural models, each trained
to predict specific aspects of the internal cellular machinery.

It relies on three main elements:

- The set of neural functions, each trained to predict a specific cellular process
  (e.g., transcription, translation, ERN-TS degradation). These networks output
  quantile functions that estimate the distribution of possible outcomes.

- A grammar, or set of rules, that defines how these neural networks are
  combined given a set of DNA constructs

- Variational embeddings of the various DNA parts, which are used by the
  neural functions to inform their predictions. These embeddings are learnt
  unsupervised, and the variational aspect allows us to make their latent space
  smooth and thus amenable to gradient-based traversal when designing new
  circuits.

We train our biomorphic networks using various sets of poly-co-transfection
experiments, each involving different transfection recipes. For each
experiment, we generate a directed acyclic graph (DAG) representing the
cellular circuit. The DAG consists of neural functions, connected according to
high-level rules reflecting cellular processes (transcription, translation,
ERN-TS degradation, etc.). Most neural functions are modulated by the
embeddings of specific DNA parts (promoter, uOrfs, ERNs), which act as
parameters that will tune the network behavior.

During training, all DAGs are optimized simultaneously, with each neural
function appearing in multiple contexts across different experiments. This
setup allows the functions to generalize their roles across varied
architectures, while the DNA embeddings learn the influence of genetic
components on specific cellular processes.

After training, we can "freeze" the neural functions and enter a design mode.
By computing gradients of the network outputs with respect to the parts
embeddings, we can traverse their smooth latent space to identify DNA that
achieve a target behavior. Finally, the embeddings are quantized to the closest
discrete DNA parts, providing designs for experimental validation.

# Modules

## Turning DNA into a composition of neural functions

### recipe.py

Handles the importing, parsing, and management of genetic circuit recipes from JSON5 files into computational representations.

### network.py

Defines the core structures for representing genetic circuits as computational graphs.

**Key classes:**

- `TranscriptionUnit` - Represents genetic elements like promoters and genes
- `Network` - Manages both the central dogma graph and compute graph
- `Slot` - Represents a position in a transcription unit that can contain parts

## Core implementation

### parameters.py

Provides a hierarchical system for managing neural network parameters with tagging and references support (allowing DAG topologies).

**Key classes:**

- `ParameterTree` - Tree-based parameter storage with tagging capabilities
- `PTree` - Base implementation of parameter trees
- `ArrayRef` - References to arrays that can span multiple locations in the tree

### compute.py

Implements the computational core for executing biomorphic neural networks. The
main idea is to enable parallelization of the execution of many different
networks by first organizing them into a stack of layers (called a `ComputeStack`)

**Key classes:**

- `ComputeStack` - Manages the execution of multiple networks in parallel
- `ComputeLayer` - Organizes nodes that can be executed in parallel
- `VirtualNode` - Represents a computational node in the network
- `ComputeConfig` - Configuration for network computation

**Key functions:**

- `build()` - Constructs the compute stack from networks
- `init()` - Initializes parameters for the stack
- `apply()` - Executes the network computation

### nodes.py

Contains implementations of neural network nodes representing biological processes.

## Training and design

### train.py + trainutils.py

Implementation of the main training loop and many training-related helper functions.

### datautils.py

Handles data processing and preparation for training. `DataManager` also holds the data for multiple networks.

**Key classes:**

- `DataManager` - Manages data for multiple networks and provides sampling utilities
- `DataRescaler` / `CompressedSymLogRescaler` - Scale data to appropriate ranges

## Plotting and Visualization

### plotting\*

Visualization tools for different plot types.

**Key modules:**

- `plotting_core.py` - Core visualization utilities
- `plotting_smooth.py` - Smoothed visualization of network behavior
- `plotting_scatter.py` - Scatter plot implementations
- `plotting_3d.py` - 3D visualization for multi-dimensional data
