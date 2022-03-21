#!/usr/bin/env python
print("Importing modules")
import networkx as nx
from networkx.generators.random_graphs import fast_gnp_random_graph
import random

print("Importing fugu")
import fugu

print("Importing Scaffold")
from fugu import Scaffold

print("Importing Bricks")
from fugu.bricks import Graph_Traversal, Vector_Input

MAX_RUNTIME = 100


def create_graph(size, p, seed):
    G = fast_gnp_random_graph(size, p, seed=seed)
    random.seed(seed)
    for (u, v) in G.edges():
        G.edges[u, v]['weight'] = random.randint(1, 10)
    return G


# Build test cases
#   test case = graph, start node, bfs tree, shortest distances to every other node
#       note: calculate the bfs and shortest path stuff later using networkx
test_cases = []
weird_case = nx.DiGraph()
weird_case.add_edge(0, 1, weight=1)
weird_case.add_edge(1, 2, weight=1)
weird_case.add_edge(2, 3, weight=1)
weird_case.add_edge(3, 4, weight=1)
weird_case.add_edge(4, 5, weight=1)
weird_case.add_edge(5, 6, weight=1)
weird_case.add_edge(6, 7, weight=1)
weird_case.add_edge(7, 8, weight=1)
weird_case.add_edge(8, 9, weight=1)
weird_case.add_edge(0, 10, weight=10)
weird_case.add_edge(1, 11, weight=9)
weird_case.add_edge(2, 12, weight=8)
weird_case.add_edge(3, 13, weight=7)
weird_case.add_edge(4, 14, weight=6)
weird_case.add_edge(5, 15, weight=5)
weird_case.add_edge(6, 16, weight=4)
weird_case.add_edge(7, 17, weight=3)
weird_case.add_edge(8, 18, weight=2)
weird_case.add_edge(9, 19, weight=1)

test_cases.append((weird_case, 0))
#test_cases.append((create_graph(10, 0.2, 3), 2))
#test_cases.append((create_graph(10, 0.2, 23), 3))
#test_cases.append((create_graph(10, 0.2, 11), 4))
#test_cases.append((create_graph(10, 0.2, 37), 5))
#test_cases.append((create_graph(10, 0.2, 19), 6))
#test_cases.append((create_graph(10, 0.2, 59), 7))

# Build bricks and scaffold
sssp_tables = []
for test_case in test_cases:
    print("---Building Scaffold---")
    sssp_scaffold = Scaffold()

    spikes = [0] * test_case[1]
    spikes.append(1)

    sssp_input = Vector_Input(spikes, coding='Raster', name='SSSPInput')
    sssp_brick = Graph_Traversal(test_case[0],
                                 name="SSSP",
                                 store_parent_info=True)
    sssp_scaffold.add_brick(sssp_input, 'input')
    sssp_scaffold.add_brick(sssp_brick, output=True)

    sssp_scaffold.lay_bricks()
    #sssp_scaffold.summary(verbose=2)

    pynn_args = {}
    pynn_args['backend'] = 'brian'
    pynn_args['verbose'] = False
    pynn_args['show_plots'] = False

    print("---Running SSSP---")
    #sssp_result = sssp_scaffold.evaluate(backend='pynn',max_runtime=MAX_RUNTIME, record_all=True, backend_args=pynn_args)
    sssp_result = sssp_scaffold.evaluate(backend='ds',
                                         max_runtime=MAX_RUNTIME,
                                         record_all=True)

    print("---Interpreting Spikes for SSSP---")
    sssp_pred = {v: -1 for v in test_case[0].nodes}
    sssp_table = {v: -1 for v in test_case[0].nodes}
    sssp_start_time = 0.0

    sssp_names = list(sssp_scaffold.graph.nodes.data('name'))
    for row in sssp_result.itertuples():
        neuron_name = sssp_names[int(row.neuron_number)][0]
        print(neuron_name, row.time)

        neuron_props = sssp_scaffold.graph.nodes[neuron_name]
        if 'begin' in neuron_name:
            sssp_start_time = row.time
        if 'is_vertex' in neuron_props:
            v = neuron_props['index'][0]
            sssp_table[v] = row.time
        if 'is_edge_reference' in neuron_props:
            u = neuron_props['from_vertex']
            v = neuron_props['to_vertex']
            sssp_pred[v] = u if u < sssp_pred[v] or sssp_pred[
                v] < 0 else sssp_pred[v]

    for v in sssp_table:
        if sssp_table[v] > -1:
            sssp_table[v] -= sssp_start_time
            sssp_table[v] /= 2.0

    final_sssp = set()
    for v in sssp_pred:
        if sssp_pred[v] > -1:
            final_sssp.add((v, sssp_pred[v]))
    sssp_tables.append((final_sssp, sssp_table))
