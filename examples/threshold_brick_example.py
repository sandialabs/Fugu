#!/usr/bin/env python
import numpy as np

print("---Importing modules---")
print("---Importing fugu---")
import fugu

print("---Importing Scaffold---")
from fugu import Scaffold

print("---Importing Bricks---")
from fugu.bricks import Threshold, Vector_Input, Copy, Dot

MAX_RUNTIME = 5
TRIALS = 200
TOLERANCE = 0.05
print("---Building test test cases---")
test_cases = []

# Params: input coding, input value, threshold, p, decay
# Expected Output: Spike %
test_cases.append((['curent', 0, 0, 1, 0], 0.0))
test_cases.append((['curent', 0.01, 0, 1, 0], 1.0))
test_cases.append((['curent', 1.00, 1, 1, 0], 0.0))
test_cases.append((['curent', 1.01, 1, 1, 0], 1.0))
test_cases.append((['curent', 1.00, 0.5, 0.75, 0], 0.75))
test_cases.append((['temporal-L', 0, 0, 1, 0], 0.0))
test_cases.append((['temporal-L', 1, 0, 1, 0], 1.0))
test_cases.append((['temporal-L', 3, 2, 1, 0], 1.0))
test_cases.append((['temporal-L', 3, 3, 1, 0], 0.0))
test_cases.append((['temporal-L', 3, 4, 1, 0], 0.0))

results = []

for parameters, answer in test_cases:
    print("---Building Scaffold---")

    #thresh_brick = Threshold()

    scaffold = Scaffold()

    scaffold.add_brick(
        Vector_Input(np.array([1]), coding='Raster', name='input1'), 'input')
    scaffold.add_brick(
        Vector_Input(np.array([0]), coding='Raster', name='input2'), 'input')
    scaffold.add_brick(Dot([parameters[1]], name='ADotOperator'),
                       (0, 0))  #don't know why i need two vector inputs
    scaffold.add_brick(Threshold(parameters[2],
                                 p=parameters[3],
                                 decay=parameters[4],
                                 name='Thresh',
                                 output_coding=parameters[0]), (2, 0),
                       output=True)

    scaffold.lay_bricks()

    #scaffold.summary(verbose=2)

    pynn_args = {}
    pynn_args['backend'] = 'brian'
    pynn_args['verbose'] = False
    pynn_args['show_plots'] = False

    #result = scaffold.evaluate(backend='pynn',max_runtime=MAX_RUNTIME, record_all=True, backend_args=pynn_args)
    evaluations = 0.0
    hits = 0.0
    print("---Running {} evaluations---".format(TRIALS))
    for i in range(TRIALS):
        result = scaffold.evaluate(backend='ds',
                                   max_runtime=MAX_RUNTIME,
                                   record_all=True)
        evaluations += 1.0

        graph_names = list(scaffold.graph.nodes.data('name'))
        for row in result.itertuples():
            neuron_name = graph_names[int(row.neuron_number)][0]
            if "Thresh" in neuron_name:
                hits += 1
                break
    #print(hits, evaluations, percentage)
    print("---Finished {} evaluations---".format(TRIALS))
    results.append(hits / evaluations)

print("---Final results---")
print(
    "Input coding, input value, threshold, p, decay, expected spike rate,actual spike rate, within {}%?"
    .format(TOLERANCE * 100))
for (params, answer), result in zip(test_cases, results):
    print("{}, {}, {}, {}, {}, {}, {}, {}".format(
        params[0], params[1], params[2], params[3], params[4], answer, result,
        abs(result - answer) < TOLERANCE))
