#!/usr/bin/env python
# -*- coding: utf-8 -*-

print("---Importing modules---")
from __future__ import print_function
import numpy as np

print("---Importing fugu---")
import fugu
from fugu import Scaffold

print("---Importing Bricks---")
from fugu import bricks

print("---Importing Backend---")
from fugu.backends import snn_Backend


class basic_AND(fugu.Brick):
    def __init__(self, name=None):
        super().__init__()
        #The brick hasn't been built yet.
        #self.is_built = False
        #Leave for compatibility, D represents the depth of the circuit.  Needs to be updated.
        self.metadata = {'D': 1}
        #We just store the name passed at construction.
        self.name = name
        #For this example, we'll let any input coding work even though the answer might not make sense.
        self.supported_codings = fugu.input_coding_types

    def build(self, graph, metadata, control_nodes, input_lists,
              input_codings):
        #Expect two inputs
        if len(input_codings) != 2:
            raise ValueError('Only two inputs supported.')
        #Keep the same coding as input 0 for the output
        #This is an arbitrary decision at this point.
        #Generally, your brick will impart some coding, but that isn't the case here.
        output_codings = [input_codings[0]]

        #All bricks should provide a neuron that spikes when the brick has completed processing.
        #We just put in a basic relay neuron that will spike when it receives any spike from its
        #single input, which is the complete_node from the first input.
        #All nodes we add to the graph should have basic neuron parameters (threshold, decay)
        #Reasonable defaults will be filled-in, but these defaults may depend on the execution platform.
        #Additionally, nodes should have a field called 'index' which is a local index used to reference the
        #position of the node.  This can be used by downstream bricks.  A simple example might be
        #a 3-bit binary representation will add 3 nodes to the graph with indices 0,1,2
        #We do have to do some work to establish best practices here.
        new_complete_node_name = self.name + '_complete'
        graph.add_node(new_complete_node_name,
                       index=-1,
                       threshold=0.0,
                       decay=0.0,
                       p=1.0,
                       potential=0.0)
        graph.add_edge(control_nodes[0]['complete'],
                       new_complete_node_name,
                       weight=1.0,
                       delay=1)

        #We also, obviously, need to build the computational portion of our graph
        and_node_name = self.name + '_0'
        graph.add_node(and_node_name,
                       index=0,
                       threshold=1.0,
                       decay=1.0,
                       p=1.0,
                       potential=0.0)
        graph.add_edge(input_lists[0][0],
                       and_node_name,
                       weight=0.75,
                       delay=1.0)
        graph.add_edge(input_lists[1][0],
                       and_node_name,
                       weight=0.75,
                       delay=1.0)
        self.is_built = True

        #Remember, bricks can have more than one output, so we need a list of list of output neurons
        output_lists = [[and_node_name]]

        return (graph, self.metadata, [{
            'complete': new_complete_node_name
        }], output_lists, output_codings)

    
print("---Building Scaffold---")

scaffold = Scaffold()
scaffold.add_brick(
    bricks.Vector_Input(np.array([1]), coding='Raster', name='Input0'),
    'input')
scaffold.add_brick(
    bricks.Vector_Input(np.array([1]), coding='Raster', name='Input1'),
    'input')
scaffold.add_brick(basic_AND(name='AND'), [0, 1], output=True)
scaffold.lay_bricks()
scaffold.summary(verbose=1)

print("---Running evaluation---")

backend = snn_Backend()
backend_args = {}
backend_args['record'] = 'all'
backend.compile(scaffold, backend_args)

result = backend.run(10)
print(result)
