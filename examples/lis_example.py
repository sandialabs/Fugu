import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import fugu
from fugu import Scaffold, Brick
from fugu.bricks import LongestIncreasingSubsequence, Vector_Input

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

#sequence = [1,4,8,6,2,7,9,3,2]
#sequence = [4,6,2,7,9]
#sequence = [1,3,2,4]
sequence = [1,4,6,2,3,7,9]
#sequence = [1,4,3,8,9]
#sequence = [1,7,4,10,8]
#sequence = [1,2,3,4,5]
#sequence = [1,3,1,2]
#sequence = [1,7,8]
LIS_brick = LongestIncreasingSubsequence(sequence, delay_alarms=False, name="LIS")

scaffold = Scaffold()
scaffold.add_brick(Vector_Input(np.array([1]), coding='Raster', name='Input0'), 'input' )
scaffold.add_brick(LIS_brick, output=True)
scaffold.lay_bricks(verbose=1)

print("Summary: ")
#print(scaffold.summary(verbose=0))
print("<<<")
result = scaffold.evaluate(backend='ds',max_runtime=50, record_all=True)

graph_names = list(scaffold.graph.nodes.data('name'))
print("time, neuron number, neuron name")
for row in result.itertuples():
    print(row.time, row.neuron_number, graph_names[int(row.neuron_number)][0])
    #print(step)
    #print(result[step])
    #names = [graph_names[i][0] for i in result[step]]
    #print("Step: {}, Spikes: {}, Spikes (named): {}".format(step, result[step], names))
