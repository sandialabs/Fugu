#!/usr/bin/env python
print("---Importing modules---")
print("---Importing fugu---")
import fugu

print("---Importing Scaffold---")
from fugu import Scaffold

print("---Importing Bricks---")
from fugu.bricks import Flow_Augmenting_Path, Vector_Input

import networkx as nx
import numpy as np

print("---Building Scaffold---")

debug = True

graph = nx.DiGraph()
graph.add_edge('s', 1, capacity=4, flow=0)
graph.add_edge('s', 2, capacity=3, flow=0)
graph.add_edge(1, 3, capacity=6, flow=0)
graph.add_edge(2, 3, capacity=6, flow=0)
graph.add_edge(2, 4, capacity=5, flow=0)
graph.add_edge(3, 't', capacity=5, flow=0)
graph.add_edge(4, 't', capacity=1, flow=0)

brick_properties = {}
brick_properties['Augmenting'] = {'flow': {}}
for edge in graph.edges():
    brick_properties['Augmenting']['flow'][edge] = 0

MAX_RUNTIME = len(graph.edges()) + 10

scaffold = Scaffold()

input_brick = Vector_Input([[0, 1]],
                           coding='Raster',
                           name='Input0',
                           time_dimension=True)
FAP_brick = Flow_Augmenting_Path(graph, name='Augmenting')

scaffold.add_brick(input_brick, 'input')
scaffold.add_brick(FAP_brick, output=True)
scaffold.lay_bricks()

pynn_args = {}
pynn_args['backend'] = 'brian'
pynn_args['verbose'] = False
pynn_args['show_plots'] = False
pynn_args['return_potentials'] = True

print("---Running evaluation---")
min_residual = 9  # max capacity

while min_residual > 0:
    print("--->Iteration")
    min_residual = 9
    result = scaffold.evaluate(
        brick_properties=brick_properties,
        backend='pynn',
        max_runtime=MAX_RUNTIME,
        record_all=True,
        backend_args=pynn_args,
    )

    scaffold.summary(verbose=2)
    spikes, potentials = result

    graph_names = list(scaffold.graph.nodes.data('name'))

    path_edges = []  # list of edges from s to t

    if debug:
        print("---spike_times---")

    recall_spikes = []
    for row in spikes.itertuples():
        neuron_name = graph_names[int(row.neuron_number)][0]
        neuron_props = scaffold.graph.nodes[neuron_name]
        if debug:
            print(neuron_name, row.time)
        if 'edge' in neuron_props:
            if neuron_props['neuron_type'] == 'recall':
                edge = neuron_props['edge']
                recall_spikes.append((row.time, edge))
    recall_spikes.reverse()
    for time, edge in recall_spikes:
        add_to_path = False
        if len(path_edges) == 0:
            if edge[1] == 't':
                add_to_path = True
        elif edge[1] == path_edges[-1][0]:
            add_to_path = True
        if add_to_path:
            path_edges.append(edge)
            flow = brick_properties['Augmenting']['flow'][edge]
            capacity = graph.edges[edge]['capacity']
            residual = capacity - flow
            if residual < min_residual:
                min_residual = residual

    if debug:
        print("aug path---")
        print(path_edges, min_residual)
    if len(path_edges) > 0:
        for edge in path_edges:
            brick_properties['Augmenting']['flow'][edge] += min_residual
    else:
        min_residual = 0

if debug:
    print("Final potentials")

max_flow = 0
final_flow_values = {}
for row in potentials.itertuples():
    neuron_name = graph_names[int(row.neuron_number)][0]
    neuron_props = scaffold.graph.nodes[neuron_name]
    if 'edge' in neuron_props:
        if neuron_props['neuron_type'] == 'capacity':
            if debug:
                print(neuron_name, row.potential, neuron_props['potential'],
                      neuron_props['threshold'])
            flow_value = neuron_props['potential'] - len(graph.edges()) - 1
            if neuron_props['edge'][1] == 't':
                max_flow += flow_value
            final_flow_values[neuron_props['edge']] = flow_value

print("Maximum flow: {}".format(max_flow))
print("Final flow values: (edge, flow value)")
for edge in graph.edges():
    print("{}, {}".format(edge, final_flow_values[edge]))
