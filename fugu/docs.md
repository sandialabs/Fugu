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
Scaffold.add_brick(self, brick_function, input_nodes=[-1], metadata=None, name=None, output=False)
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
    + record_all - bool value to indicate if all neurons spikes are to be recorded (Default: False)

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
Brick.build(self, graph, metadata, control_nodes, input_lists, input_codings)
```

Build the computational graph of the brick. Method must be defined in any class inheriting from Brick.

Arguments:
    + graph - networkx graph
    + metadata - A dictionary of shapes and parameters
    + control_nodes - list of dictionary of auxillary nodes.  Acceptable keys include: 'complete' - A list of neurons that fire when the brick is done, 'begin' - A list of neurons that fire when the brick begins computation (used for temporal processing)
    + input_lists - list of lists of nodes for input neurons
    + input_codings - list of input coding types (as strings)

## InputBrick
```python
InputBrick(self)
```
Abstract Base class for handling inputs inherited from Brick
### get_input_value
```python
InputBrick.get_input_value(self, t=None)
```

Abstract method to get input values. InputBriacks must implement this method

Arguments:
    + t - type of input (Default: None)

## Spike_Input
```python
Spike_Input(self, spikes, time_dimension=False, coding='Undefined', name=None)
```
Class to handle Spike Input. Inherits from InputBrick
### build
```python
Spike_Input.build(self, graph, metadata, control_nodes, input_lists, input_codings)
```

Build spike input brick.

Arguments:
    + graph - networkx graph to define connections of the computational graph
    + metadata - dictionary to define the shapes and parameters of the brick
    + control_nodes - list of dictionary of auxillary nodes.  Excpected keys: 'complete' - A list of neurons that fire when the brick is done
    + input_lists - list of nodes that will contain input
    + input_coding - list of input coding formats

Returns:
    + graph of a computational elements and connections
    + dictionary of output parameters (shape, coding, layers, depth, etc)
    + list of dictionary of control nodes ('complete')
    + list of output
    + list of coding formats of output

## Threshold
```python
Threshold(self, threshold, decay=0.0, p=1.0, name=None, output_coding=None)
```
Class to handle Threshold Brick. Inherits from Brick
### build
```python
Threshold.build(self, graph, metadata, control_nodes, input_lists, input_codings)
```

Build Threshold brick.

Arguments:
    + graph - networkx graph to define connections of the computational graph
    + metadata - dictionary to define the shapes and parameters of the brick
    + control_nodes - list of dictionary of auxillary nodes.
      Expected keys: 'complete' - A neurons that fire when the brick is done
                     'begin' - A neurons that first when the brick begins processing (for temporal coded inputs)
    + input_lists - list of nodes that will contain input
    + input_coding - list of input coding formats

Returns:
    + graph of a computational elements and connections
    + dictionary of output parameters (shape, coding, layers, depth, etc)
    + list dictionary of control nodes ('complete')
    + list of output
    + list of coding formats of output

## Dot
```python
Dot(self, weights, name=None)
```
Class to handle the Dot brick. Inherits from Brick
### build
```python
Dot.build(self, graph, metadata, control_nodes, input_lists, input_codings)
```

Build Dot brick.

Arguments:
    + graph - networkx graph to define connections of the computational graph
    + metadata - dictionary to define the shapes and parameters of the brick
    + control_nodes - list of dictionary of auxillary nodes.  Excpected keys: 'complete' - A list of neurons that fire when the brick is done
    + input_lists - list of nodes that will contain input
    + input_coding - list of input coding formats.  ('Raster', 'Undefined' supported)

Returns:
    + graph of a computational elements and connections
    + dictionary of output parameters (shape, coding, layers, depth, etc)
    + list dictionary of control nodes ('complete')
    + list of output edges
    + list of coding formats of output ('current')

## Copy
```python
Copy(self, name=None)
```
Class to handle Copy Brick. Inherits from Brick

### build
```python
Copy.build(self, graph, metadata, control_nodes, input_lists, input_codings)
```

Build Copy brick.

Arguments:
    + graph - networkx graph to define connections of the computational graph
    + metadata - dictionary to define the shapes and parameters of the brick
    + control_nodes - list of dictionaries of auxillary nodes.  Excpected keys: 'complete' - A list of neurons that fire when the brick is done
    + input_lists - list of nodes that will contain input
    + input_coding - list of input coding formats

Returns:
    + graph of a computational elements and connections
    + dictionary of output parameters (shape, coding, layers, depth, etc)
    + list of dictionaries of control nodes ('complete')
    + list of output
    + list of coding formats of output

## AND_OR
```python
AND_OR(self, mode='AND', name=None)
```
Brick for perform a logical AND/OR.  Operation is performed entry-wise, matching based on index.  All codings are supported.
Arguments:
    + mode - Either 'And' or 'Or'; determines the operation
+ name - Name of the brick.  If not specified, a default will be used.  Name should be unique.

### build
```python
AND_OR.build(self, graph, metadata, control_nodes, input_lists, input_codings)
```

Build AND_OR brick.

Arguments:
    + graph - networkx graph to define connections of the computational graph
    + metadata - dictionary to define the shapes and parameters of the brick
    + control_nodes - dictionary of lists of auxillary networkx nodes.  Excpected keys: 'complete' - A list of neurons that fire when the brick is done
    + input_lists - list of nodes that will contain input
    + input_coding - list of input coding formats.  All codings are allowed.

Returns:
    + graph of a computational elements and connections
    + dictionary of output parameters (shape, coding, layers, depth, etc)
    + dictionary of control nodes ('complete')
    + list of output (1 output)
    + list of coding formats of output (Coding matches input coding)

## Shortest_Path_Length
```python
Shortest_Path_Length(self, target_graph, target_node, name=None, output_coding='temporal-L')
```
This brick provides a single-source shortest path length determination.


### build
```python
Shortest_Path_Length.build(self, graph, metadata, control_nodes, input_lists, input_codings)
```

Build Parity brick.

Arguments:
    + graph - networkx graph to define connections of the computational graph
    + metadata - dictionary to define the shapes and parameters of the brick
    + control_nodes - dictionary of lists of auxillary networkx nodes.  Excpected keys: 'complete' - A list of neurons that fire when the brick is done
    + input_lists - list of nodes that will contain input
    + input_coding - list of input coding formats.  All coding types supported

Returns:
    + graph of a computational elements and connections
    + dictionary of output parameters (shape, coding, layers, depth, etc)
    + dictionary of control nodes ('complete')
    + list of output
    + list of coding formats of output

## ParityCheck
```python
ParityCheck(self, name=None)
```
Brick to compute the parity of a 4 bit input.
The output spikes after 2 time steps if the input has odd parity

### build
```python
ParityCheck.build(self, graph, metadata, control_nodes, input_lists, input_codings)
```

Build Parity brick.

Arguments:
    + graph - networkx graph to define connections of the computational graph
    + metadata - dictionary to define the shapes and parameters of the brick
    + control_nodes - dictionary of lists of auxillary networkx nodes.  Excpected keys: 'complete' - A list of neurons that fire when the brick is done
    + input_lists - list of nodes that will contain input
    + input_coding - list of input coding formats

Returns:
    + graph of a computational elements and connections
    + dictionary of output parameters (shape, coding, layers, depth, etc)
    + dictionary of control nodes ('complete')
    + list of output
    + list of coding formats of output

