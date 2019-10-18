#!/usr/bin/env python
print("Importing modules")
import networkx as nx
from networkx.generators.random_graphs import fast_gnp_random_graph
import random
from collections import deque

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
#GRAPH_SIZE = 2 ** 4 + 8
GRAPH_SIZE = 2 ** 7

def create_graph(size, p, seed):
    # @TODO: Need to replace this with a Kronecker/R-Mat generator
    G = fast_gnp_random_graph(size, p, seed=seed)
    for (u,v) in G.edges():
        weight = random.randint(1,20)
        G.edges[u,v]['weight'] = weight
    return G

def estimate_max_time(graph, key):
    # Given a graph, estimate the maximum runtime needed for BFS and SSSP
    # BFS
    bfs_layers = 1
    visited = {node:False for node in graph.nodes}
    levels = {node:0 for node in graph.nodes}
    queue = deque()
    queue.append(key)
    levels[key] = 1
    while len(queue) > 0:
        top = queue.popleft()
        visited[top] = True
        for neighbor in graph.neighbors(top):
            if not visited[neighbor]:
                levels[neighbor] = levels[top] + 1
                queue.append(neighbor)

    max_level = 0
    for v in levels:
        max_level = levels[v] if levels[v] > max_level else max_level

    # SSSP
    _, dist = nx.dijkstra_predecessor_and_distance(graph, key)
    max_dist = 0
    for key in dist:
        max_dist = dist[key] if dist[key] > max_dist else max_dist
    
    return max_level, max_dist 

random.seed(3)
graph = create_graph(GRAPH_SIZE, 0.3, 3)

results = {}
timing_results = {}
metrics = {}

pynn_args = {}
pynn_args['backend'] = 'spinnaker'
pynn_args['collect_metrics'] = True


# Generate graph
for case_index in range(1):
    search_key = random.randint(1,GRAPH_SIZE)

    print("---Building BFS Scaffold---")
    bfs_scaffold = Scaffold()

    spikes = [0] * search_key
    spikes[-1] = 1

    bfs_brick = Breadth_First_Search(graph, name="BFS", store_edge_references=True)
    bfs_input = Vector_Input(spikes, coding='Raster', name='BFSInput')
    bfs_scaffold.add_brick(bfs_input, 'input')
    bfs_scaffold.add_brick(bfs_brick, output=True)

    bfs_scaffold.lay_bricks()
    if DEBUG:
        bfs_scaffold.summary(verbose=2)

    if DEBUG:
        pynn_args['verbose'] = True
    else:
        pynn_args['verbose'] = False 
    pynn_args['show_plots'] = False

    print("---Running BFS---")
    MAX_RUNTIME = 2 * GRAPH_SIZE * GRAPH_SIZE * 10 + 10
    max_level, max_dist = estimate_max_time(graph, search_key)

    pynn_args['scale_factor'] = 1.0
    bfs_result = bfs_scaffold.evaluate(backend=BACKEND,max_runtime=3 * max_level + GRAPH_SIZE, record_all=True, backend_args=pynn_args)
    bfs_runtime = bfs_scaffold.metrics['runtime']
    bfs_embedding_time =  bfs_scaffold.metrics['embed_time']

    print("---Interpreting Spikes for BFS---")
    bfs_pass = True
    bfs_finished = False

    bfs_pred = {v:-1 for v in graph.nodes}
    bfs_levels = {}
    bfs_names = list(bfs_scaffold.graph.nodes.data('name'))

    curr_level = 0
    curr_time = 0.0
    bfs_spikes = 0
    last_bfs_spike_time = 0.0
    for row in bfs_result.sort_values("time").itertuples():
        if DEBUG:
            print(neuron_name, row.time)
        bfs_spikes += 1
        neuron_name = bfs_names[int(row.neuron_number)][0]

        neuron_props = bfs_scaffold.graph.nodes[neuron_name]
        if 'is_vertex' in neuron_props:
            if row.time > curr_time:
                curr_level += 1
                curr_time = row.time
            vertex = neuron_props['index'][0]
            if vertex in bfs_levels:
                bfs_pass = False
                print("Detected cycle: {}".format(vertex))
                break
            else:
                bfs_levels[vertex] = curr_level
        if 'is_edge_reference' in neuron_props:
            u = neuron_props['from_vertex']
            v = neuron_props['to_vertex']
            if bfs_pred[v] < 0:
                bfs_pred[v] = u
        if 'complete' in neuron_name:
            bfs_finished = True

        last_bfs_spike_time = row.time

    if DEBUG:
        print("preds>>>")
        for v in bfs_pred:
            print(v,bfs_pred[v])
        print("levels>>>")
        for v in bfs_levels:
            print(v, bfs_levels[v])

    if not bfs_finished:
        print("BFS did not finish")

    for u in bfs_pred:
        v = bfs_pred[u]
        if v > -1:
            u_level = bfs_levels[u]
            v_level = bfs_levels[v]
            if abs(u_level - v_level) != 1:
                bfs_pass = False
                print("Levels between node and pred != 1: {} {}".format(u,v))
                break
            if (u,v) not in graph.edges() and (v,u) not in graph.edges():
                bfs_pass = False
                break

    for edge in graph.edges():
        u = edge[0]
        v = edge[1]
        u_level = bfs_levels[u]
        v_level = bfs_levels[v]
        if abs(u_level - v_level) > 1:
            bfs_pass = False
            print("Levels between edge > 1: {} {}".format(*edge))
            break

    print("---Building SSSP Scaffold---")
    sssp_scaffold = Scaffold()

    sssp_brick = Shortest_Path(graph, name="SSSP", return_path=True)
    sssp_input = Vector_Input(spikes, coding='Raster', name='SSSPInput')
    sssp_scaffold.add_brick(sssp_input, 'input')
    sssp_scaffold.add_brick(sssp_brick, output=True)

    sssp_scaffold.lay_bricks()
    if DEBUG:
        sssp_scaffold.summary(verbose=2)
    if DEBUG:
        pynn_args['verbose'] = True
    else:
        pynn_args['verbose'] = False 

    print("---Running SSSP---")
    pynn_args['scale_factor'] = 1.0
    sssp_result = sssp_scaffold.evaluate(backend=BACKEND,max_runtime=2 * max_dist + GRAPH_SIZE, record_all=True, backend_args=pynn_args)
    sssp_runtime = sssp_scaffold.metrics['runtime']
    sssp_embedding_time = sssp_scaffold.metrics['embed_time']

    print("---Interpreting Spikes for SSSP---")
    sssp_pass = True
    sssp_finished = False

    sssp_pred = {v:-1 for v in graph.nodes}
    sssp_table = {v:-1 for v in graph.nodes}
    sssp_start_time = 0.0

    sssp_names = list(sssp_scaffold.graph.nodes.data('name'))
    sssp_spikes = 0
    last_sssp_spike_time = 0.0
    for row in sssp_result.itertuples():
        if DEBUG:
            print(neuron_name, row.time)
        sssp_spikes += 1
        neuron_name = sssp_names[int(row.neuron_number)][0]

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
        if 'complete' in neuron_name:
            sssp_finished = True
        last_sssp_spike_time = row.time

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

    if not sssp_finished:
        print("SSSP did not finish")

    for u in sssp_pred:
        v = sssp_pred[u]
        if v > -1:
            u_dist = sssp_table[u]
            v_dist = sssp_table[v]
            if u_dist == -1 or v_dist == -1:
                sssp_pass = False
                print("Distance and pred don't match: {} {}".format(u,v))
                break
            else:
                edge_weight = graph.get_edge_data(u,v)['weight']
                if abs(u_dist - v_dist) > edge_weight:
                    sssp_pass = False
                    print("Distance larger than tree edge weight ({},{}): {} {} {}".format(u,v,u_dist,v_dist, edge_weight))
                    break

    for u,v in graph.edges():
        u_dist = sssp_table[u]
        v_dist = sssp_table[v]
        edge_weight = graph.get_edge_data(u,v)['weight']
        if abs(u_dist - v_dist) > edge_weight:
            sssp_pass = False
            print("Distance larger than edge weight ({},{}): {} {} {}".format(u,v,u_dist,v_dist, edge_weight))
            break


    results[search_key] = (bfs_pass, sssp_pass, bfs_embedding_time, sssp_embedding_time, bfs_runtime, sssp_runtime, last_bfs_spike_time, last_sssp_spike_time, bfs_spikes, sssp_spikes, len(bfs_scaffold.graph.nodes()), len(sssp_scaffold.graph.nodes()))

print("Case ID, bfs pass, sssp pass, bfs_embed_time, sssp_embed_time, bfs_runtime, sssp_runtime, last_bfs_spike, last_sssp_spike, bfs_spikes, sssp_spikes, bfs_circuit_size, sssp_circuit_size")
for key in results:
    print(("{}, "+", ".join(["{}" for r in results[key]])).format(key, *results[key]))
