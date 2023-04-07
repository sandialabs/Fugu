Copyright 2022 National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Fugu
A python library for computational neural graphs.

Fugu is a high-level framework specifically designed for developing spiking circuits in terms of computation graphs. Accordingly, with a base leaky-integrate-and fire (LIF) neuron model at its core, neural circuits are built as *bricks*. These foundational computations are then combined and composed as scaffolds to construct larger computations. This allows us to describe spiking circuits in terms of neural features common to most NMC architectures rather than platform specific designs. In addition to architectural abstraction, the compositionality concept of Fugu not only facilitates a hierarchical approach to functionality development but also enables adding pre and post processing operations to overarching neural circuits. Such properties position Fugu to help explore under what parameterization or scale a neural approach may offer an advantage. For example, prior work has analyzed neural algorithms for computational kernels like sorting, optimization, and graph analytics identifying different regions in which a neural advantage exists accounting for neural circuit setup, timing, or other factors.

# Install

## Dependencies
A full list of dependencies is listed in requirements.txt.  The high level dependencies are:

- Numpy
- NetworkX
- Pandas

### A Note on Running Examples

Some of the examples additionally require Jupyter and matplotlib.


## Using Pip
```
git clone http://github.com/SNL-NERL/Fugu.git
cd Fugu
pip install -r requirements.txt
pip install --upgrade .
```

# Documentation
Documentation is currently spread across several files.  We are working on including docstrings on all the classes and methods.

For now, you can check:
- This README.md
- The examples folder
- [API Documentation](https://snl-nerl.github.io/Fugu/build/html/index.html)

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
