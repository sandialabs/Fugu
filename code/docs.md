# fugu

## Scaffold
```python
Scaffold(self)
```
Class to handle a scaffold of bricks
### supported_backends
Built-in mutable sequence.

If no argument is given, the constructor creates a new empty list.
The argument must be an iterable if specified.
### add_brick
```python
Scaffold.add_brick(self, brick_function, input_nodes=[-1], dimensionality=None, name=None, output=False)
```

Add a brick to the scaffold.

Arguments:
    + brick_function - object of type brick
    + input_nodes - list of node numbers (Default: [-1])
    + dimesionality -  dictionary of shapes and parameters of the brick (Default: None)
    + name - string of the brick's name (Default: none)
    + output - bool flag to indicate if a brick is an output brick (Default: False)

Returns:
    + None

Exceptions:
    + Raises ValueError if node name is already used.

### all_nodes_built
```python
Scaffold.all_nodes_built(self, verbose=0)
```

Check if all nodes are built.

Arguments:
    + verbose - int to indicate level of verbosity (Default: 0 to indicate no messages)
Returns:
    + bool with True if all nodes are built, Fase otherwise

### all_in_neighbors_built
```python
Scaffold.all_in_neighbors_built(self, node)
```

Check if all neighbors of a node are built.

Arguments:
    + node - node whose neighbors are checked

Returns:
    + bool - indicates if all neighbors are built.

### lay_bricks
```python
Scaffold.lay_bricks(self, verbose=0)
```

Build a computational graph that can be used by the backend.

Arguments:
    + verbose - int value to specify level of verbosity (Default: 0 to indicate None)

Returns:
    networkX diGraph

### evaluate
```python
Scaffold.evaluate(self, max_runtime=10, backend='ds', record_all=False)
```

Run the computational graph through the backend.

Arguments:
    + max_runtime - int value to specify number of time steps (Default: 10)
    + backend - string value of the backend simulator or device name (Default: 'ds')
    + record_all - bool value to indicate if all neurons spikesa re to be recorded (Default: False)

Returns:
    + dictionary of time step and spiking neurons. (if record_all is True, all spiking neurons are shown
    else only the output neurons)

Exceptions:
    + ValueError if backend is not in list of supported backends

### summary
```python
Scaffold.summary(self)
```

Display a summary of the scaffold.

## Brick
```python
Brick(self)
```
Abstract Base Class definition of a Brick class
### build
```python
Brick.build(self, graph, dimensionality, complete_node, input_lists, input_codings)
```
Build the computational graph of the brick
## InputBrick
```python
InputBrick(self)
```
Abstract Base class for handling inputs inherited from Brick
## Spike_Input
```python
Spike_Input(self, spikes, time_dimension=False, coding='Undefined', name=None)
```

### build
```python
Spike_Input.build(self, graph, dimensionality, complete_node, input_lists, input_codings)
```

Build spike input brick.

Arguments:
    + graph - networkx graph to define connections of hte computational graph
    + dimensionality - dictionary to define the dimensionality of the brick
    + complete_node - list of networkx nodes to indicate completion of the computation
    + input_lists - list of nodes that will contain input
    + input_coding - list of input coding formats

Returns:
    + graph of a computational elements and connections
    + dictionary of output parameters (shape, coding, layers, depth, etc)
    + list of complete nodes
    + list of output
    + list of coding formats of output

## ParityCheck
```python
ParityCheck(self, name=None)
```
Brick to compute the parity of a 4 bit input.
The output spikes after 2 time steps if the input has odd parity

author: Srideep Musuvathy
email: smusuva@sandia.gov
last updated: April 8, 2019
