#!/usr/bin/env python
import numpy as np

print("---Importing modules---")
print("---Importing fugu---")
import fugu

print("---Importing Scaffold---")
from fugu.scaffold import Scaffold

print("---Importing Bricks---")
from fugu.bricks import Vector_Input, Register, Max

print("---Importing Backend---")
from fugu.backends import snn_Backend

MAX_RUNTIME = 605

backend = 'ds'
pynn_args = {}
pynn_args['backend'] = 'brian'
pynn_args['verbose'] = False
pynn_args['show_plots'] = False
pynn_args['store_voltage'] = False

scaffold = Scaffold()

scaffold.add_brick(
    Vector_Input(np.array([0, 1, 0, 1]), coding='Raster', name='input1'),
    'input')
scaffold.add_brick(
    Vector_Input(np.array([0, 0, 1, 0]), coding='Raster', name='input2'),
    'input')
scaffold.add_brick(
    Vector_Input(np.array([1, 0, 1, 0]), coding='Raster', name='input3'),
    'input')
scaffold.add_brick(
    Vector_Input(np.array([1, 0, 1, 1]), coding='Raster', name='input4'),
    'input')
scaffold.add_brick(Max(name='MaxBrick'),
                   input_nodes=[(0, 0), (1, 0), (2, 0), (3, 0)],
                   output=True)

scaffold.lay_bricks()

scaffold.summary(verbose=2)

backend = snn_Backend()
backend_args = {}
backend_args['record'] = 'all'
backend.compile(scaffold, backend_args)

print("---Running evaluation---")
result = backend.run(100)

print("---Finished evaluation---")

graph_names = list(scaffold.graph.nodes.data('name'))
for row in result.sort_values('time').itertuples():
    neuron_name = graph_names[int(row.neuron_number)][0]
    print(neuron_name, row.time)
