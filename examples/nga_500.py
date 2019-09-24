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
from fugu.bricks import Breadth_First_Search, Shortest_Path, Vector_Input

# NGA 500 outline:
#   - Generate graph
#   - Embed graph onto hardware (collect metrics)
#   - Generate 64 search keys
#       - degree must be >= 1 (does not include self-loops)
#   - Compute BFS predessor array (collect metrics)
#       - verify tree
#   - Compute SSSP predessor array and distance array (collect metrics)
#       - verify tree

MAX_RUNTIME = 2000
DEBUG = True

def create_graph(size, p, seed):
    G = fast_gnp_random_graph(size, p, seed=seed)
    random.seed(seed)
    for (u,v) in G.edges():
        weight = random.randint(1,10)
        G.edges[u,v]['weight'] = weight
    return G

graph = create_graph(64, 0.3, 3)

results = []

# Generate graph
for case_index in range(1):
    print("---Building Scaffold---")
    search_key = random.randint(1,64)
    bfs_scaffold = Scaffold()
    sssp_scaffold = Scaffold()

    spikes = [0] * search_key
    spikes.append(1)

    bfs_brick = Breadth_First_Search(graph, name="BFS")
    bfs_input = Vector_Input(spikes, coding='Raster', name='BFSInput')
    bfs_scaffold.add_brick(bfs_input, 'input')
    bfs_scaffold.add_brick(bfs_brick, output=True)

    bfs_scaffold.lay_bricks()
    #bfs_scaffold.summary(verbose=2)

    sssp_input = Vector_Input(spikes, coding='Raster', name='SSSPInput')
    sssp_brick = Shortest_Path(graph, name="SSSP", return_path=True)
    sssp_scaffold.add_brick(sssp_input, 'input')
    sssp_scaffold.add_brick(sssp_brick, output=True)

    sssp_scaffold.lay_bricks()
    #sssp_scaffold.summary(verbose=2)

    pynn_args = {}
    pynn_args['backend'] = 'spinnaker'
    pynn_args['verbose'] = False
    pynn_args['show_plots'] = False

    print("---Running BFS---")
    MAX_RUNTIME = 2 * 64 * 10

    bfs_result = bfs_scaffold.evaluate(backend='pynn',max_runtime=MAX_RUNTIME, record_all=True, backend_args=pynn_args)
    #bfs_result = bfs_scaffold.evaluate(backend='ds',max_runtime=MAX_RUNTIME, record_all=True, backend_args=pynn_args)

    print("---Interpreting Spikes for BFS---")
    bfs_pred = {v:-1 for v in graph.nodes}
    bfs_names = list(bfs_scaffold.graph.nodes.data('name'))
    for row in bfs_result.itertuples():
        neuron_name = bfs_names[int(row.neuron_number)][0]

        neuron_props = bfs_scaffold.graph.nodes[neuron_name]
        if 'is_edge_reference' in neuron_props:
            u = neuron_props['from_vertex']
            v = neuron_props['to_vertex']
            bfs_pred[v] = u if u < bfs_pred[v] or bfs_pred[v] < 0 else bfs_pred[v]

    final_bfs = set()
    for v in bfs_pred:
        if bfs_pred[v] > -1:
            final_bfs.add((v, bfs_pred[v]))

    bfs_pred_pass = True
    expected_bfs_preds = list(nx.bfs_predecessors(graph, source=search_key))
    if DEBUG:
        print("expected bfs (pred):")
        print(expected_bfs_preds)
        print("actual:")
        print(final_bfs)

    if len(expected_bfs_preds) != len(final_bfs):
        bfs_pred_pass = False
        if DEBUG:
            print("BFS Pred lists are different lengths")
    else:
        for pair in expected_bfs_preds:
            if pair not in final_bfs:
                bfs_pred_pass = False
                if DEBUG:
                    print("{} not in final_bfs".format(pair))
                break

    print("---Running SSSP---")
    #sssp_result = sssp_scaffold.evaluate(backend='pynn',max_runtime=MAX_RUNTIME, record_all=True, backend_args=pynn_args)
    sssp_result = sssp_scaffold.evaluate(backend='ds',max_runtime=MAX_RUNTIME, record_all=True, backend_args=pynn_args)

    print("---Interpreting Spikes for SSSP---")
    sssp_pred = {v:-1 for v in graph.nodes}
    sssp_table = {v:-1 for v in graph.nodes}
    sssp_start_time = 0.0

    sssp_names = list(sssp_scaffold.graph.nodes.data('name'))
    for row in sssp_result.itertuples():
        neuron_name = sssp_names[int(row.neuron_number)][0]
        #print(neuron_name, row.time)

        neuron_props = sssp_scaffold.graph.nodes[neuron_name]
        if 'begin' in neuron_name:
            sssp_start_time = row.time
        if 'is_vertex' in neuron_props:
            v = neuron_props['index'][0]
            sssp_table[v] = row.time
        if 'is_edge_reference' in neuron_props:
            u = neuron_props['from_vertex']
            v = neuron_props['to_vertex']
            sssp_pred[v] = u if u < sssp_pred[v] or sssp_pred[v] < 0 else sssp_pred[v]

    for v in sssp_table:
        if sssp_table[v] > -1:
            sssp_table[v] -= sssp_start_time
            sssp_table[v] /= 2.0

    final_sssp = set()
    for v in sssp_pred:
        if sssp_pred[v] > -1:
            final_sssp.add((v, sssp_pred[v]))


    sssp_pred_pass = True
    sssp_dist_pass = True
    expected_tables = nx.dijkstra_predecessor_and_distance(graph, search_key)

    if DEBUG:
        print("expected sssp (pred):")
        print(expected_tables[0])
        print("actual:")
        print(final_sssp)

    if len(expected_tables[0].keys()) - 1 != len(final_sssp):
        sssp_pass = False
    else:
        for v in expected_tables[0]: 
            pred = expected_tables[0][v]
            if len(pred) > 0:
                if (v, pred[0]) not in final_sssp:
                    sssp_pred_pass = False
                    if DEBUG:
                        print("({},{}) not in final_sssp".format(v, pred[0]))
                    break

    if DEBUG:
        print("expected sssp (dist):")
        print(expected_tables[1])
        print("actual:")
        print(sssp_table)

    keys = sssp_table.keys()
    for v in expected_tables[1]:
        expected_value = expected_tables[1][v]
        actual_value = sssp_table[v]
        if expected_value != actual_value:
            sssp_dist_pass = False
            if DEBUG:
                print("{} does not match {}".format(actual_value, expected_value))
            break
        keys.remove(v)

    for v in keys:
        if sssp_table[v] > -1:
            sssp_dist_pass = False
            if DEBUG:
                print("Distance is not -1 for {}".format(v))
            break

    results.append((search_key, bfs_pred_pass, sssp_pred_pass, sssp_dist_pass))

print("Case ID, bfs pred pass, sssp pred pass, sssp dist pass")
for result in results:
    print("{}, {}, {}, {}".format(*result))
