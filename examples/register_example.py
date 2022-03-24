#!/usr/bin/env python
import numpy as np

print("---Importing modules---")
print("---Importing fugu---")
import fugu

print("---Importing Scaffold---")
from fugu.scaffold import Scaffold

print("---Importing Bricks---")
from fugu.bricks import Vector_Input, Register

print("---Importing Backend---")
from fugu.backends import snn_Backend


MAX_RUNTIME = 605


scaffold = Scaffold()

inputs = [[0] * MAX_RUNTIME for i in range(2)]

inputs[0][200] = 1  
spikes = [5, 20, 45]
for spike in spikes:
    inputs[1][spike] = 1

scaffold.add_brick(
    Vector_Input(np.array(inputs),
                 coding='Raster',
                 name='input',
                 time_dimension=True), 'input')
scaffold.add_brick(Register(5, name='register1'),
                   input_nodes=(0,0),
                   output=True)
scaffold.add_brick(Register(5, name='register2'),
                   input_nodes=(0,0),
                   output=True)

scaffold.lay_bricks()

scaffold.summary(verbose=2)

print("---Running evaluation---")
backend = snn_Backend()
backend_args = {}
backend_args['record'] = 'all'
backend.compile(scaffold, backend_args)
result = backend.evaluate(MAX_RUNTIME)

print("---Finished evaluation---")

graph_names = list(scaffold.graph.nodes.data('name'))
for row in result.sort_values('time').itertuples():
    neuron_name = graph_names[int(row.neuron_number)][0]
    print(neuron_name, row.time)
