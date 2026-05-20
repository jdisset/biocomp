The biocompiler is a comprehensive machine learning framework for neuromorphic
synthetic biology circuits. It consists of multiple interconnected modules that
enable the design, training, and prediction of cell behavior given cellular
circuits they are transfected with. It uses a novel architecture called
biomorphic neural networks (BNNs) where compositions of specialized neural
functions represent cellular processes. The framework is designed for high
computational efficiency, parallel training, and extensive metadata tracking
for reproducibility.

Unlike standard neural networks, **Biomorphic Neural Networks (BNNs)** map
directly to biological processes. A key feature of these circuits is the use of
**Endoribonucleases (ERNs)**--specific proteins (like CasE or Csy4) that
recognize and cleave specific RNA hairpins. This cleavage suppresses the
translation of downstream genes, providing the essential non-linear inhibitory
mechanism required for complex logic and neuromorphic behavior.

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

## General Overview

The codebase uses a **Graph Rewriting System** to translate biological intent into computation. The system starts with a "Central Dogma Graph" representing the physical flow of genetic information (DNA $\to$ RNA $\to$ Protein). It then iteratively applies declarative rewriting rules (defined in `biorules.py`) to transform this biological topology into a computational graph. This graph is finally compiled into a **Compute Stack**--a flattened, JAX-based structure optimized for highly parallel execution, allowing the simulation and training of thousands of different genetic circuit variations simultaneously.

## Module Descriptions

#### 1\. Biological Data & Recipe Definition

- **`biocomp.recipe`**: This is the user-facing API for describing experiments.
  - **`Recipe`**: The top-level container for a set of experimental conditions.
  - **`CoTransfection`**: Represents a specific mixture of plasmids introduced into a cell. It handles `ratios` (stoichiometry) and `fluo_bias` (experimental noise/bias).
  - **`TranscriptionUnit` (TU)**: Represents a single gene on a DNA strand, composed of a list of **`Slot`s**.
  - **`Slot`**: A placeholder for a biological part. It can hold a specific part name (e.g., "hEF1a") or a list of options (e.g., `["1x_uORF", "2x_uORF"]`) for design optimization.
- **`biocomp.library`**: Manages the database of biological parts.
  - **`PartsLibrary`**: Loads part definitions from a SQL/SQLite database (`models.py`). It resolves hierarchical part definitions (L0 basic parts assembling into L1 genes, assembling into L2 plasmids).
  - It maintains the registry of **Sequestrons** (the specific ERN/Target pairs) and their types.
- **`biocomp.part_embeddings`**: Defines the mapping from discrete biological parts to continuous vectors. For example, it maps categorical promoter names to learnable "transcription rate" vectors used by the neural network.

#### 2\. Graph Engine (The Core Transformation Logic)

- **`biocomp.network`**: The orchestrator of the build process.
  - **`build_central_dogma_graph_direct`**: Converts a `Recipe` into a raw graph of biological entities (DNA nodes connected to RNA nodes, connected to Protein nodes).
  - **`recipe_to_networks`**: The main entry point. It takes a recipe, builds the biological graph, applies the rewriting rules to create a compute graph, and optionally inverts it (for design tasks).
- **`biocomp.graphengine`**: A generic graph manipulation framework.
  - **`GraphState`**: An immutable snapshot of the network containing `nodes` and `edges`.
  - **`graphs_are_isomorphic`**: A utility to verify that two different recipes produce the exact same computational topology.
- **`biocomp.graphrules`**: Implements the Domain Specific Language (DSL) for graph rewriting.
  - **`MatchQuery`**: Allows searching the graph for specific topologies (e.g., "Find all RNA nodes connected to a Translation node").
  - **`GraphRewritingRule`**: pairs a query with **`Action`s** (like `AddNode`, `RewireEdges`, `DeleteNode`).
- **`biocomp.biorules`**: The specific biological logic.
  - It defines rules like `create_aggregation_nodes` (to model plasmids mixing in a cell) or `sequestron_ERN` rules (identifying where an ERN protein creates a feedback loop by cleaving a specific RNA).

#### 3\. Computation & JAX Engine

- **`biocomp.compute`**: The execution engine.
  - **`ComputeStack`**: Takes multiple `Network` graphs and "stacks" distinct nodes into layers (`StackLayer`). This allows the system to execute the `transcription` step for 100 different networks in a single JAX operation.
  - **`_generate_apply_method`**: Compiles the entire stack into a single, JIT-compilable JAX function.
- **`biocomp.nodes`**: The mathematical implementations of biological steps.
  - **`transcription` / `translation`**: Neural transformations (MLPs) that map inputs (upstream concentrations + part embeddings) to outputs (downstream concentrations).
  - **`sequestron_ERN`**: Models the Endoribonuclease non-linearity. It calculates the inhibition of translation based on the affinity between the ERN protein and the target RNA.
  - **`aggregation`**: Computes the weighted sum of plasmid outputs based on transfection ratios.
- **`biocomp.parameters`**: A hierarchical parameter manager.
  - **`ParameterTree`**: A dictionary-like structure compatible with JAX pytrees. It supports "tagging" (e.g., separating `shared` biological constants from `local` experiment-specific variables).
  - **`ArrayRef`**: A mechanism allowing multiple nodes in the compute graph to reference the same underlying parameter memory (crucial for sharing part embeddings across the stack).

#### 4\. Training, Design & Utils

- **`biocomp.train`**: The training loop.
  - **`sorting_loss`**: A specialized loss function. Since flow cytometry data is a distribution, this loss encourages the network to learn the _quantiles_ of the fluorescence distribution rather than just the mean.
- **`biocomp.design`**: Solves the inverse problem.
  - **`DesignManager`**: Given a target behavior (e.g., a logic gate truth table or a complex curve drawn in an SVG), it freezes the biological constants and optimizes the _input parameters_ (promoters, uORFs) to match the target.
- **`biocomp.quantization`**: Implements "Variational Quantization." During design, the system explores a continuous "embedding space" for parts. This module handles snapping those continuous values back to the nearest valid discrete part (e.g., "Promoter A" vs "Promoter B") while maintaining differentiability via the Straight-Through Estimator.

#### 5\. Visualization

Visualization tools for different plot types.

**Key submodules:**

- `plotting_core.py` - Core visualization utilities
- `plotting_smooth.py` - Smoothed visualization of network behavior
- `plotting_scatter.py` - Scatter plot implementations
- `plotting_3d.py` - 3D visualization for multi-dimensional data

# Contribution Guidelines

## Coding Style

**DRY (Don't Repeat Yourself)**:

- Extract common patterns into functions
- Use comprehensions over loops
- Centralize configuration and constants

**SOLID Principles**:

- Single Responsibility: Each function/class does one thing
- Open/Closed: Extend via rules/configuration, not modification
- Liskov Substitution: Consistent interfaces
- Interface Segregation: Minimal public APIs
- Dependency Inversion: Depend on abstractions (e.g GraphState) not implementations

**Terse and Elegant**:

- Prioritize clarity, modularity and maintainability, but try really hard to be concise. NO "ENTERPRISE CODE".
- Always prefer generic solutions to hardcoded special cases, but avoid over-engineering. What we want is pragmatic elegance.
- Minimize lines of code while preserving readability. Code golf within reason. Don't be afraid of some amount of cleverness, but of course avoid obfuscation.
- Always reread and refactor code to make it more concise after writing it. There's always some fat to trim.

"One day I will find the right words, and they will be simple." - Jack Kerouac

## Comment Style

**Extremely terse, informal**:

- No comments for obvious code
- No capitalization of first letter for inline comments
- Only comment non-obvious intent or biological context
- Basically, avoid comments. If you feel the need to comment, consider rewriting the code for clarity instead.

## Testing Style

**Functional pytest**:

- No test classes, just functions with fixtures
- Parameterized tests for combinatorial coverage
- Direct assertions, no verbose messages
- Fixture reuse across test files via imports
- Avoid special cases like the plague: if a fix requires manual intervention for specific test data, it's probably the wrong fix.

- Example:

```python
@pytest.fixture
def lib():
    return load_lib()

def test_something(lib, simple_single_reporter):
    with LibraryContext.with_library(lib):
        # some setup ...

        # a bunch of assertions:
        assert ...
```

## Linters and Formatters to run:

Always use ruff + basedpyright for linting, type checking, and formatting.

`ruff check filename.py`
`basedpyright --pythonpath /opt/homebrew/Caskroom/miniconda/base/envs/py311/bin/python filename.py`
`ruff format filename.py`
