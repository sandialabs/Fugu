#!/usr/bin/env python
print("Importing modules")
import networkx as nx
import numpy as np

print("Importing fugu")
import fugu
print("Importing Scaffold")
from fugu import Scaffold
print("Importing Bricks")
from fugu.bricks import TemporalAdder, Vector_Input


print("Building test sequences")
test_cases = []

test_cases.append(([10, 7],17))
test_cases.append(([10, 8],18))
test_cases.append(([6, 8],14))
test_cases.append(([9, 9],18))
test_cases.append(([1, 9],10))

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

    scaffold.add_brick(Vector_Input(np.array(time_vector), coding='Raster', name='Input', time_dimension=True), 'input')


    #scaffold.add_brick(Vector_Input(np.array([[0,0,1]]), coding='Raster', name='Input0', time_dimension = time_dim), 'input' )
    #scaffold.add_brick(Vector_Input(np.array([[0, 10]]), coding='Raster', name='Input1', time_dimension = time_dim), 'input' )

    scaffold.add_brick(adder_brick, output=True)

    scaffold.lay_bricks()

    pynn_args = {}
    pynn_args['backend'] = 'brian'
    pynn_args['verbose'] = False 
    pynn_args['show_plots'] = False 

    print("---Running evaluation---")

    max_time = (answer+4) * 2
    #result = scaffold.evaluate(backend='pynn',max_runtime=max_time, record_all=True, backend_args=pynn_args)
    #result = scaffold.evaluate(backend='ds', max_runtime=max_time, record_all=True)
    result = scaffold.evaluate(backend='snn', max_runtime=max_time, record_all=True)

    graph_names = list(scaffold.graph.nodes.data('name'))
    print("---Finished evaluation:---")
    for row in result.itertuples():
        neuron_name = graph_names[int(row.neuron_number)][0]
        #print(neuron_name, row.time)
        if 'Sum' in neuron_name:
            results.append(0.5 * row.time - 3)
        #print(neuron_name, row[-1])

print("---Final results---")
print("sequence,expected,actual")
for (sequence, answer), result in zip(test_cases, results):
    print("{}, {}, {}".format(sequence, answer, result))
    #print("Sequence: {}".format(sequence))
    #print("Expected answer: {}".format(answer))
    #print("Actual answer: {}".format(result))
