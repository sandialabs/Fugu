#!/usr/bin/env python
print("---Importing modules---")
print("---Importing fugu---")
import fugu
print("---Importing Scaffold---")
from fugu import Scaffold
print("---Importing Bricks---")
from fugu.bricks import LIS, Vector_Input

MAX_RUNTIME = 500
print("---Building test test cases---")
test_cases = []

test_cases.append(([1,4,8,6,2,7,9,3,2],5))
test_cases.append(([1,4,8,6,2,7,19,13,14],6))
test_cases.append(([1,5,6,7,5,3,4,13,14],6))
test_cases.append(([4,6,2,7,9],4))
test_cases.append(([1,4,6,2,3,7,9],5))
test_cases.append(([1,7,4,5,8],4))
test_cases.append(([1,2,3,4],4))
test_cases.append(([5,9,5,7],2))
test_cases.append(([1,3,1,2],2))
test_cases.append(([1,2],2))
test_cases.append(([2,1],1))
test_cases.append(([20,21],2))
test_cases.append(([20,20, 20, 1,1],1))

results = []

for sequence, answer in test_cases:

    print("---Building Scaffold---")

    LIS_brick = LIS(len(sequence), name="LIS")

    scaffold = Scaffold()
    num_in_sequence = len(sequence)
    max_time = max(sequence)
    spike_times = [[0] * (max_time + 1) for i in range(num_in_sequence)]
    for i, time in enumerate(sequence):
        spike_times[i][time] = 1
    #print(spike_times)
    #for time in sequence:
        #spike_times.append([0] * time)
        #spike_times[-1].append(1)

    #print(spike_times)
    #for i, spike_array in enumerate(spike_times):
        #print("i, spike_array {} {}".format(i, spike_array))
        #scaffold.add_brick(Vector_Input(spike_array, coding='Raster', name='Input{}'.format(i)), 'input')

    scaffold.add_brick(Vector_Input(spike_times, coding='Raster', name='Input0', time_dimension=True), 'input')

    scaffold.add_brick(LIS_brick, output=True)
    scaffold.lay_bricks()

    #scaffold.summary(verbose=2)

    pynn_args = {}
    pynn_args['backend'] = 'brian'
    pynn_args['verbose'] = False 
    pynn_args['show_plots'] = False

    print("---Running evaluation---")

    #result = scaffold.evaluate(backend='pynn',max_runtime=MAX_RUNTIME, record_all=True, backend_args=pynn_args)
    result = scaffold.evaluate(backend='ds',max_runtime=MAX_RUNTIME, record_all=True)

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
for (sequence, answer), result in zip(test_cases, results):
    print("{}, {}, {}, {}".format(sequence, answer, result, answer == result))
