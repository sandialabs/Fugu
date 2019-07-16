import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import fugu
from fugu import Scaffold, Brick
from fugu.bricks import LIS, Vector_Input

def check_for_spikes(neuron_name, scaffold):
    #Get the names for all the neurons
    #Then find the output neuron
    output_neuron_index = [node for node in scaffold.graph.nodes].index(neuron_name)
    found_a_spike = False
    when = None
    for timestep in result:
        if output_neuron_index in result[timestep]:
            found_a_spike=True
            when = timestep
    return found_a_spike, when

test_sequences = []

#test_sequences.append(([1,4,8,6,2,7,9,3,2],5))
#test_sequences.append(([1,4,8,6,2,7,19,13,14],6))
#test_sequences.append(([4,6,2,7,9],4))
#test_sequences.append(([1,4,6,2,3,7,9],5))
test_sequences.append(([1,7,4,5,8],4))
#test_sequences.append(([1,2,3,4],4))
#test_sequences.append(([1,3,1,2],2))
test_sequences.append(([1,2],2))
#test_sequences.append(([1],1))

for sequence, answer in test_sequences:
    LIS_brick = LIS(sequence, name="LIS")

    scaffold = Scaffold()
    scaffold.add_brick(Vector_Input(np.array([1]), coding='Raster', name='Input0'), 'input' )
    scaffold.add_brick(LIS_brick, output=True)
    scaffold.lay_bricks()

    #print("Summary: ")
    #print(scaffold.summary(verbose=2))
    #print("<<<")
    result = scaffold.evaluate(backend='pynn',max_runtime=50, record_all=True)

    graph_names = list(scaffold.graph.nodes.data('name'))
    print("---Finished evaluation:---")
    #print("Spikes table:")
    #print("time, neuron number, neuron name")
    lis = 0
    for row in result.itertuples():
        neuron_name = graph_names[int(row.neuron_number)][0]
        #print(row.time, row.neuron_number, neuron_name)
        if "L_" in neuron_name:
            level = int(neuron_name.split("-")[0][2:])
            if level > lis:
                lis = level
    #print("---")

    print("Expected answer: {}".format(answer))
    print("Actual answer: {}".format(lis))
