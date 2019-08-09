#!/usr/bin/env python
print("Importing modules")
import networkx as nx
import numpy as np

print("Importing fugu")
import fugu
print("Importing Scaffold")
from fugu import Scaffold
print("Importing Bricks")
from fugu.bricks import LIS, Vector_Input


print("Building test sequences")
test_sequences = []

test_sequences.append(([1,4,8,6,2,7,9,3,2],5))
test_sequences.append(([1,4,8,6,2,7,19,13,14],6))
test_sequences.append(([4,6,2,7,9],4))
test_sequences.append(([1,4,6,2,3,7,9],5))
test_sequences.append(([1,7,4,5,8],4))
test_sequences.append(([1,2,3,4],4))
test_sequences.append(([1,3,1,2],2))
test_sequences.append(([1,2],2))
test_sequences.append(([1],1))

results = []

for sequence, answer in test_sequences:

    print("---Building Scaffold---")

    LIS_brick = LIS(sequence, name="LIS")
    scaffold = Scaffold()
    scaffold.add_brick(Vector_Input(np.array([1]), coding='Raster', name='Input0'), 'input' )
    scaffold.add_brick(LIS_brick, output=True)
    scaffold.lay_bricks()

    pynn_args = {}
    pynn_args['backend'] = 'spinnaker'
    pynn_args['verbose'] = False 

    print("---Running evaluation---")

    result = scaffold.evaluate(backend='pynn',max_runtime=50, record_all=True, backend_args=pynn_args)

    graph_names = list(scaffold.graph.nodes.data('name'))
    print("---Finished evaluation:---")
    lis = 0
    for row in result.itertuples():
        neuron_name = graph_names[int(row.neuron_number)][0]
        #print(neuron_name, row.time)
        if "Main" in neuron_name:
            level = int(neuron_name.split("_")[1])
            if level > lis:
                lis = level
    results.append(lis)
print("---Final results---")
print("sequence,expected,actual")
for (sequence, answer), result in zip(test_sequences, results):
    print("{}, {}, {}".format(sequence, answer, result))
    #print("Sequence: {}".format(sequence))
    #print("Expected answer: {}".format(answer))
    #print("Actual answer: {}".format(result))
