# Fugu Ports

Fugu has two kinds of users: 
1. those who build applications using bricks, 
2. and those who create new bricks. 

This document provides information for both kinds of users.


## Ports for Regular Users

Building a Fugu application consists mainly of creating and hooking up bricks. The job of a port is to represent an input or output of a brick. They allow you to hook these up using a human-readable name.

Here is an example of a simple circuit. The code below never mentions ports, because the default values are sufficient.

```python
import fugu
from fugu.scaffold import *
from fugu.bricks import *
import numpy as np

S = Scaffold()
I1 = S.add_brick(Vector_Input(np.array([1,0,1,0]), coding='Raster', name='input1'))
I2 = S.add_brick(Vector_Input(np.array([1,1,0,0]), coding='Raster', name='input2'))
A = S.add_brick(AND_OR(), output=True)
S.connect(I1, A)
S.connect(I2, A)
```

Here is the same example with the ports specified explicitly:

```python
S.connect(I1, A, 'output', 'input')
S.connect(I2, A, 'output', 'input')
```

The most common case is that a brick has only one input port and one output port, and those are named `input` and `output`. This happens to be the case in the example above. The Scaffold knows this, so the user can often omit those parameters.

Each brick class can list documentation about its ports. This is possible without creating an instance of the brick.

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

This says that the `AND_OR` brick has one input port and one output port, ingeniously named `input` and `output`. The port named `input` is capable of taking exactly 2 inputs (minimum=maximum=2). In the circuit example above, `I1` and `I2` are both connected to `input`. This is mandatory. Any more or less than 2 inputs would result in an error. Scaffold automatically converts `input` into two ports named `input1` and `input2` that receive the actual connections from `I1` and `I2`.

It is possible to refer to ports by number rather than name. The `index` entry in the documentation gives this value.

Under each port is a list of channels. At its heart, a port is a way for the scaffold to connect one group of neurons to another. A port often has several different groups that are related to each other, but not exactly the same. Channels are a way to organize these groups. In the simple example above, the `data` channel carries the main population of neurons, while `complete` carries a single neuron that signals when data from one brick is ready for the next one to use.

For two ports to be fully compatible, they need to have the same set of channels, and each matching pair of channels needs to have at least one coding scheme in common. If a channel does not list any coding scheme then it is compatible with all of them.


## Ports for Developers

When developing a new brick, you need to deal with two aspects of ports: 
1. document what ports are available on the brick
2. work with the information passed into `build2()` by the scaffold.

### Documenting ports

The `Brick` class specifies two class methods, `input_ports()` and `output_ports()`. Each of these return a dictionary that maps from port names to specification objects. These are essentially reflection methods. They give a machine-readable description of the brick's interface. The function `show_ports()` simply converts this to a human-readable form on the console.

All the relevant port-description classes are brought in when you `import fugu.scaffold`. As you develop your brick, you should define the `input_ports()` and `output_ports()` functions. Here is an example from the `AND_OR` brick:

```python
@classmethod
def input_ports(cls) -> dict[str, PortSpec]:
    port = PortSpec(name='input', minimum=2, maximum=2)
    port.channels['data']     = ChannelSpec(name='data', coding=input_coding_types)
    port.channels['complete'] = ChannelSpec(name='complete')
    return {port.name: port}
```

The `output_ports()` method is very similar. Each method works by first creating a `PortSpec` object, then filling in the channel dictionary inside `PortSpec`. Finally, it assembles the dictionary and returns it to the caller. A brick class with multiple would do basically the same thing, just with additional entries.

### Working with ports during `build2()`

At the top of `build2()`, first unpack the inputs provided by the scaffold. These come in as dictionary from port name to a port data object. The port data objects are supplied by the output of other brick `build2()` calls, and scaffold assembles the dictionary using the port mappings set up by the user. The keys in the dictionary are the input port names declared by your brick. The usual procedure is to retrieve each port into a local variable. You can do this by explicit name:

```python
def build2(self, graph, inputs: dict[str, PortData] = {}):
    input1 = inputs['input1']
```

Alternately, if this is a port that can have multiple input connections, you can use a utility function to retrieve a tupal of ports:
```python
input1, input2 = PortUtil.get_autoports(inputs, 'input', 2)
```

After collecting the inputs, set up the output ports. It is helpful to define these before building neural populations, so they can be added to the output as you go. The simplest way to prepare a port output structure is to build it from the bricks's port description. There is a utility function for this:
```python
result = PortUtil.make_ports_from_specs(AND_OR.output_ports())
```

The variable `result` is destined to be output by `build2()`. Next, is is useful to unpack specific channels, as these are where the neurons are stored:
```python
output = result['output']
data = output.channels['data']
```

Then fill in `data` with whatever output neurons the brick creates. A similar process applies to the signal `complete`. Finally, return the object:
```python
# build neuron population(s)
data.neurons.append(neuron)
...
return result
```