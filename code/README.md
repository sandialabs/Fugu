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
- complete_node: A list of neurons that transmits a 'done' signal (Generally one from each input)
- input_lists: A list of lists of input neurons.  Each neuron is marked with a local index used for encodings.
- input_codings: A list of types of input codings.  See input_coding_types

Output:
A tuple (graph, dimensionality, complete_node, output_lists, output_codings)
- graph: Graph that is being built
- dimensionality: Dictionary containing relevant dimensionality information
- complete_node: A list of neurons that transmists a 'done' signal (Generally one for each output)
- output_lists: A list of lists of output neurons.  Each neuron is marked with a local index used for encodings.
- output_codings: A list of types of codings.  See input_coding_types


## Known Bugs and todos
- Inputs that take both pre-computed values and input spikes will not lay correctly.
- Nodes only accept input from the first output of an input node.
- `_create_ds_injection` only works on vector-shaped input
- `_create_ds_injection` only works on input layers with 1 input (this may be okay)
- Input handling needs to be re-written.  Currently relies on fragile order of nodes.  Additionally, we should support streaming data.
= Many checks are missing
- Mismatch between input and output layer namings.  Bricks should be able to be input and/or hidden and/or output layers.
- Maximum runtime should be determined by the depth of the graph.
- Delay brick needs to be updated as well as conversion bricks
- `Scaffold.resolve_timing` needs to be re-tested with new Delay brick
- Dimensionliaty dictionary should be removed
- Inputs/Outputs should be bundled as a list of dictionaries rather than parallel lists
- Plotting functions likely do not and wll need to be re-tested
- Bricks in brick_1.py are likely broken and will need to be fixed.
