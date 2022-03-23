#!/usr/bin/env python
print("---Importing modules---")
import networkx as nx
import numpy as np

print("---Importing Fugu---")
import fugu
from fugu import Scaffold

print("---Importing Bricks---")
from fugu.bricks import TemporalAdder, Vector_Input

print("---Importing Backend---")
from fugu.backends import snn_Backend


print("---Building test sequences---")
test_cases = []
test_cases.append(([10, 7],17))
test_cases.append(([10, 8],18))
test_cases.append(([6, 8],14))
test_cases.append(([9, 9],18))
test_cases.append(([1, 9], 10))

results = []

for spike_times, answer in test_cases:

    print("---Building Scaffold---")

    adder_brick = TemporalAdder(len(spike_times), name="Adder")
    scaffold = Scaffold()
    time_dim = True

    max_time = max(spike_times)
    time_vector = [[0] * (2 * (max_time + 1)) for i in spike_times]
    time_vector[0][spike_times[0] * 2] = 1
    time_vector[1][spike_times[1] * 2] = 1

    scaffold.add_brick(
        Vector_Input(np.array(time_vector),
                     coding='Raster',
                     name='Input',
                     time_dimension=True), 'input')


    scaffold.add_brick(adder_brick, input_nodes=[-1], output=True)

    scaffold.lay_bricks()

    backend = snn_Backend()
    backend_args = {}
    backend_args['record'] = 'all'
    backend.compile(scaffold, backend_args)

    print("---Running evaluation---")

    max_time = 1000
    result = backend.run(max_time)

    graph_names = list(scaffold.graph.nodes.data('name'))
    print("---Finished evaluation---")
    for row in result.itertuples():
        neuron_name = graph_names[int(row.neuron_number)][0]
        if 'Sum' in neuron_name:
            results.append(0.5 * row.time - 3)

print("---Final results---")
print("sequence,expected,actual")
for (sequence, answer), result in zip(test_cases, results):
    print("{}, {}, {}".format(sequence, answer, result))
