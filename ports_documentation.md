# Fugu Ports

Fugu has two types of users:
1. Those who build applications using bricks.
2. Those who create new bricks.

This document provides information for both user groups.

## Ports for Regular Users

Building a Fugu application involves creating and connecting bricks. Ports represent the inputs and outputs of bricks, allowing connections using human-readable names.

### Example: Simple Circuit

Here is an example of a simple circuit. The default port values are sufficient, so ports are not explicitly mentioned:

```python
import fugu
from fugu.scaffold import *
from fugu.bricks import *
import numpy as np

S = Scaffold()
I1 = S.add_brick(Vector_Input(np.array([1, 0, 1, 0]), coding='Raster', name='input1'))
I2 = S.add_brick(Vector_Input(np.array([1, 1, 0, 0]), coding='Raster', name='input2'))
A = S.add_brick(AND_OR(), output=True)
S.connect(I1, A)
S.connect(I2, A)
```

Here is the same example with ports explicitly specified:

```python
S.connect(I1, A, 'output', 'input')
S.connect(I2, A, 'output', 'input')
```

Most bricks have one input port and one output port, named `input` and `output`. In such cases, the Scaffold allows users to omit port parameters.

### Viewing Port Documentation

Each brick class can list documentation about its ports without requiring an instance of the brick:

```python
dir(fugu.bricks)
['AND_OR', 'Addition', ... 'Copy', 'Delay', 'Dot', ...]

AND_OR.show_ports()
Inputs:
  'input'
    index   = 0
    minimum = 2
    maximum = 2
    channels:
      'data'
        Required
        coding = ['current', 'unary-B', ... 'Rate', 'Undefined']
      'complete'
        Required
Outputs:
  'output'
    index   = 0
    channels:
      'data'
        coding = ['current', 'unary-B', ... 'Rate', 'Undefined']
      'complete'
```

```python
Vector_Input.show_ports()
No inputs
Outputs:
  'output'
    index   = 0
    channels:
      'data'
      'begin'
      'complete'
```

### Port Details

The `AND_OR` brick has one input port (`input`) and one output port (`output`). The `input` port requires exactly two connections (minimum=maximum=2). In the example above, `I1` and `I2` are connected to `input`. Any deviation from the required number of connections results in an error.

Scaffold automatically converts `input` into two ports (`input1` and `input2`) to handle the connections.

Ports can also be referred to by their index, as specified in the `index` entry of the documentation.

### Channels

Ports connect groups of neurons, often organized into multiple related groups called channels. For two ports to be compatible:
1. They must have the same set of channels.
2. Each matching channel pair must share at least one coding scheme.

If a channel does not list any coding schemes, it is compatible with all schemes.

---

## Ports for Developers

When developing a new brick, you need to:
1. Document the brick's ports.
2. Handle port information passed into `build2()` by the scaffold.

### Documenting Ports

The `Brick` class provides two class methods: `input_ports()` and `output_ports()`. These methods return dictionaries mapping port names to specification objects, which describe the brick's interface. The `show_ports()` function converts this information into a human-readable format.

Here is an example from the `AND_OR` brick:

```python
@classmethod
def input_ports(cls) -> dict[str, PortSpec]:
    port = PortSpec(name='input', minimum=2, maximum=2)
    port.channels['data']     = ChannelSpec(name='data', coding=input_coding_types)
    port.channels['complete'] = ChannelSpec(name='complete')
    return {port.name: port}
```

The `output_ports()` method follows a similar structure. For bricks with multiple ports, additional entries are added to the dictionary.

### Handling Ports in `build2()`

At the start of `build2()`, unpack the inputs provided by the scaffold. These inputs are passed as a dictionary mapping port names to port data objects. Retrieve each port into a local variable:

```python
def build2(self, graph, inputs: dict[str, PortData] = {}):
    input1 = inputs['input1']
```

For ports with multiple connections, use a utility function to retrieve a tuple of ports:

```python
input1, input2 = PortUtil.get_autoports(inputs, 'input', 2)
```

#### Setting Up Output Ports

Define output ports before building neural populations. Use the utility function `make_ports_from_specs()` to create the output structure:

```python
result = PortUtil.make_ports_from_specs(AND_OR.output_ports())
```

Unpack specific channels to store neurons:

```python
output = result['output']
data = output.channels['data']
```

Populate the channels with neurons created by the brick:

```python
# Build neuron population(s)
data.neurons.append(neuron)
...
return result
```
