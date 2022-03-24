# Fugu
A python library for computational neural graphs.

# Install
```
git clone https://cee-gitlab.sandia.gov/nerl/Fugu.git
cd Fugu
pip install -r requirements.txt
pip install --upgrade .
```

An alternative way to install for development (using anaconda environments, python3)
```
conda create -n fugu anaconda
conda activate fugu
git clone git@cee-gitlab.sandia.gov:nerl/Fugu.git
cd Fugu
pip install -r requirements.txt
pip install -e .
```

# Fugu Notes

Please see the provided example(s).

## Scaffold

The `Scaffold` object is a graph that contains bricks at each node.  In reality, the Scaffold is only responsible for the organization of bricks.  All functionality is held in the bricks themselves.

### Scaffold.add_brick(self, brick_function, input_nodes=[-1], dimensionality = default_brick_dimensionality, name=None)

## Bricks

Each `Brick` represents one computational function.  Bricks are attached to a Scaffold.  Bricks have certain key properties:

- dimensionality:  A dictionary containing information such as the input and output sizes, circuit depth (if defined), and the types of codings.
- self.supported_codings:  A list of supported codings for this brick. A complete list of codings is avialable at input_coding_types .   
- is_built:  A simple boolean saying whether or not the brick as been built
- name: A string representing the brick

### Brick.build(self, graph, dimensionality, complete_node, input_lists, input_codings)

This function forms the section of the graph corresponding to the brick.

Input parameters:
- graph: graph that is being built
- dimensionality: dictionary containing relevant dimensionality information
- control_nodes: A list of dictionaries of neurons that transmit control signals. A 'done' signal (Generally one from each input) is included in `control_nodes[i]['complete']`.
- input_lists: A list of lists of input neurons.  Each neuron is marked with a local index used for encodings.
- input_codings: A list of types of input codings.  See input_coding_types

Output:
A tuple (graph, dimensionality, complete_node, output_lists, output_codings)
- graph: Graph that is being built
- dimensionality: Dictionary containing relevant dimensionality information
- complete_node: A list of neurons that transmists a 'done' signal (Generally one for each output)
- output_lists: A list of lists of output neurons.  Each neuron is marked with a local index used for encodings.
- output_codings: A list of types of codings.  See input_coding_types

### Details on `control_nodes`
We've seen through experience that it can be extremely helpful to have neurons relay control
information between bricks.  These neurons should be included in the `control_nodes`.  Regular
rules apply to these neurons (e.g. naming must be globally unique and indices are locally unique).

`control_nodes` is a *list* of *dictionaries*.  Each entry in the list corresponds with
an input (if calling `Brick.build`) or an output (if returning from `Brick.build`).

| Key | Required | Description |
| ------ | ------ | ------ |
| 'complete' | All Inputs/Outputs | A neuron that fires when a brick is done processing. |
| 'begin' | Temporally-coded Inputs/Outputs | A neuron that fires when a brick begins providing output. |

## Documentation
Documentation is currently spread across several files.  We are working on including docstrings on all the classes and methods.

For now, you can check:
- This README.md
- api.md
- docs.md
- Ipython notebook examples


## Dependencies
A full list of dependencies is listed in requirements.txt.  The high level dependencies are:

- Numpy
- Scipy
- NetworkX
- Pandas
