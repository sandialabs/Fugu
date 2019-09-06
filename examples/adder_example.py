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
#test_cases.append(([10, 8],18))

results = []

for spike_times, answer in test_cases:

    print("---Building Scaffold---")

    adder_brick = TemporalAdder(len(spike_times), name="Adder")
    scaffold = Scaffold()
    time_dim = True 

    index = 0
    for time in spike_times:
        time_array = [0 for i in range(time)]
        time_array.append(1)
        time_array[5] = 1
        scaffold.add_brick(Vector_Input(np.array([time_array, time_array]), coding='Raster', name='Input{}'.format(index), time_dimension = time_dim), 'input' )
        index += 1

    #scaffold.add_brick(Vector_Input(np.array([[0,0,1]]), coding='Raster', name='Input0', time_dimension = time_dim), 'input' )
    #scaffold.add_brick(Vector_Input(np.array([[0, 10]]), coding='Raster', name='Input1', time_dimension = time_dim), 'input' )

    scaffold.add_brick(adder_brick, [0, 1], output=True)
    scaffold.lay_bricks()
    #scaffold.summary(verbose=4)
    #print(scaffold.circuit.nodes[0]['brick'].vector)

    pynn_args = {}
    pynn_args['backend'] = 'brian'
    pynn_args['verbose'] = True 

    print("---Running evaluation---")

    result = scaffold.evaluate(backend='pynn',max_runtime=100, record_all=True, backend_args=pynn_args)
    #result = scaffold.evaluate(backend='ds', max_runtime=100, record_all=True)

    graph_names = list(scaffold.graph.nodes.data('name'))
    print("---Finished evaluation:---")
    for row in result.itertuples():
        neuron_name = graph_names[int(row.neuron_number)][0]
        print(neuron_name, row)
        if 'output' in neuron_name:
            results.append(row[-1] - 3)
        #print(neuron_name, row[-1])

print("---Final results---")
print("sequence,expected,actual")
for (sequence, answer), result in zip(test_cases, results):
    print("{}, {}, {}".format(sequence, answer, result))
    #print("Sequence: {}".format(sequence))
    #print("Expected answer: {}".format(answer))
    #print("Actual answer: {}".format(result))
