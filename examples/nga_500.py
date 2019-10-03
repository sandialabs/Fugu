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
BACKEND = "pynn"
DEBUG = False 
GRAPH_SIZE = 2 ** 6

def create_graph(size, p, seed):
    # @TODO: Need to replace this with a Kronecker/R-Mat generator
    G = fast_gnp_random_graph(size, p, seed=seed)
    random.seed(seed)
    for (u,v) in G.edges():
        weight = random.randint(1,10)
        G.edges[u,v]['weight'] = weight
    return G

graph = create_graph(GRAPH_SIZE, 0.3, 3)

results = []

# Generate graph
for case_index in range(16):
    print("---Building Scaffold---")
    search_key = random.randint(1,GRAPH_SIZE)
    bfs_scaffold = Scaffold()
    sssp_scaffold = Scaffold()

    spikes = [0] * search_key
    spikes.append(1)

    bfs_brick = Breadth_First_Search(graph, name="BFS", store_edge_references=True)
    bfs_input = Vector_Input(spikes, coding='Raster', name='BFSInput')
    bfs_scaffold.add_brick(bfs_input, 'input')
    bfs_scaffold.add_brick(bfs_brick, output=True)

    bfs_scaffold.lay_bricks()
    #bfs_scaffold.summary(verbose=2)

    sssp_brick = Shortest_Path(graph, name="SSSP", return_path=True)
    sssp_input = Vector_Input(spikes, coding='Raster', name='SSSPInput')
    sssp_scaffold.add_brick(sssp_input, 'input')
    sssp_scaffold.add_brick(sssp_brick, output=True)

    sssp_scaffold.lay_bricks()
    #sssp_scaffold.summary(verbose=2)

    pynn_args = {}
    pynn_args['backend'] = 'brian'
    pynn_args['verbose'] = False
    pynn_args['show_plots'] = False

    print("---Running BFS---")
    MAX_RUNTIME = 2 * 64 * 10

    bfs_result = bfs_scaffold.evaluate(backend=BACKEND,max_runtime=MAX_RUNTIME, record_all=True, backend_args=pynn_args)

    print("---Interpreting Spikes for BFS---")
    bfs_pass = True

    bfs_pred = {v:-1 for v in graph.nodes}
    bfs_levels = {}
    bfs_names = list(bfs_scaffold.graph.nodes.data('name'))

    curr_level = 0
    curr_time = 0.0
    for row in bfs_result.sort_values("time").itertuples():
        neuron_name = bfs_names[int(row.neuron_number)][0]

        neuron_props = bfs_scaffold.graph.nodes[neuron_name]
        if 'is_vertex' in neuron_props:
            if row.time > curr_time:
                curr_level += 1
                curr_time = row.time
            vertex = neuron_props['index'][0]
            if vertex in bfs_levels:
                bfs_pass = False
                if DEBUG:
                    print("Detected cycle: {}".format(vertex))
                break
            else:
                bfs_levels[neuron_props['index'][0]] = curr_level

        if 'is_edge_reference' in neuron_props:
            u = neuron_props['from_vertex']
            v = neuron_props['to_vertex']
            bfs_pred[v] = u if u < bfs_pred[v] or bfs_pred[v] < 0 else bfs_pred[v]
        if DEBUG:
            print(neuron_name, row.time)

    if DEBUG:
        print("preds>>>")
        for v in bfs_pred:
            print(v,bfs_pred[v])
        print("levels>>>")
        for v in bfs_levels:
            print(v, bfs_levels[v])

    for u in bfs_pred:
        v = bfs_pred[u]
        if v > -1:
            u_level = bfs_levels[u]
            v_level = bfs_levels[v]
            if abs(u_level - v_level) != 1:
                bfs_pass = False
                if DEBUG:
                    print("Levels between node and pred != 1: {} {}".format(u,v))
                break
            if (u,v) not in graph.edges() and (v,u) not in graph.edges():
                bfs_pass = False
                break

    for edge in graph.edges():
        u = edge[0]
        v = edge[1]
        if bfs_pred[u] == v or bfs_pred[v] == u:
            u_level = bfs_levels[u]
            v_level = bfs_levels[v]
            if abs(u_level - v_level) != 1:
                bfs_pass = False
                if DEBUG:
                    print("Levels between edge != 1: {} {}".format(*edge))
                break

    print("---Running SSSP---")
    sssp_result = sssp_scaffold.evaluate(backend=BACKEND,max_runtime=MAX_RUNTIME, record_all=True, backend_args=pynn_args)

    print("---Interpreting Spikes for SSSP---")
    sssp_pass = True

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

    if DEBUG:
        print(sssp_pred)
        print(sssp_table)
        print("Original weights:>>>")
        for edge in graph.edges():
            print("{}: weight {}".format(edge, graph.get_edge_data(*edge)['weight']))

    for u in sssp_pred:
        v = sssp_pred[u]
        if v > -1:
            u_dist = sssp_table[u]
            v_dist = sssp_table[v]
            if u_dist == -1 or v_dist == -1:
                sssp_pass = False
                if DEBUG:
                    print("Distance and pred don't match: {} {}".format(u,v))
                break
            else:
                edge_weight = graph.get_edge_data(u,v)['weight']
                if abs(u_dist - v_dist) > edge_weight:
                    sssp_pass = False
                    if DEBUG:
                        print("Distance larger than tree edge weight ({},{}): {} {} {}".format(u,v,u_dist,v_dist, edge_weight))
                    break

    for u,v in graph.edges():
        if sssp_pred[u] == v or sssp_pred[v] == u:
            u_dist = sssp_table[u]
            v_dist = sssp_table[v]
            edge_weight = graph.get_edge_data(u,v)['weight']
            if abs(u_dist - v_dist) > edge_weight:
                sssp_pass = False
                if DEBUG:
                    print("Distance larger than edge weight ({},{}): {} {} {}".format(u,v,u_dist,v_dist, edge_weight))
                break


    results.append((search_key, bfs_pass, sssp_pass))

print("Case ID, bfs pass, sssp pass")
for result in results:
    print("{}, {}, {}".format(*result))
