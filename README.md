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
# Documentation
Documentation is currently spread across several files.  We are working on including docstrings on all the classes and methods.

For now, you can check:
- This README.md
- The docs folder
- The examples folder

# Basic concepts

## Scaffold

The `Scaffold` object is a graph that contains bricks at each node.  In reality, the Scaffold is only responsible for the organization of bricks.  All functionality is held in the bricks themselves.


## Bricks

Each `Brick` represents one computational function.  Bricks are attached to a Scaffold.  Bricks have certain key properties:

- metadata:  A dictionary containing information such as the input and output sizes, circuit depth (if defined), and the types of codings.
- upported_codings:  A list of supported codings for this brick. A complete list of codings is avialable at input_coding_types .   
- is_built:  A simple boolean saying whether or not the brick as been built
- name: A string representing the brick

### Brick.build(self, graph, metadata, complete_node, input_lists, input_codings)

This function forms the section of the graph corresponding to the brick.

Input parameters:
- graph: graph that is being built
- metadata: dictionary containing relevant dimensionality information
- control_nodes: A list of dictionaries of neurons that transmit control signals. A 'done' signal (Generally one from each input) is included in `control_nodes[i]['complete']`.
- input_lists: A list of lists of input neurons.  Each neuron is marked with a local index used for encodings.
- input_codings: A list of types of input codings.  See input_coding_types

Output:
A tuple (graph, metadata, complete_node, output_lists, output_codings)
- graph: Graph that is being built
- metadata: Dictionary containing relevant metadata information
- control_nodes: A dictionary of lists of neurons that transmists control information (see below) 
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




## Dependencies
A full list of dependencies is listed in requirements.txt.  The high level dependencies are:

- Numpy
- Scipy
- NetworkX
- Pandas

## Documentation
To generate documentation on your home computer, download sphinx.
```
pip install -U Sphinx 
```
or 
```
conda install Sphinx
```
Navigate to the Fugu/docs folder.  Use the sphinx-build html command to generate the html files on your computer.
```
sphinx-build -b html -a source/ build/html 
```
The documentation will be in docs/build/html 
Open introduction.html.  From this page you can you can navigate through the full website, Fugu Module, etc.
