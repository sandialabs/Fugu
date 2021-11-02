import networkx as nx
import pandas as pd
import numpy as np
from collections import deque


def results_df_from_dict(results_dictionary, key, value):
    df = pd.DataFrame()
    for k in results_dictionary:
        df2 = pd.DataFrame()
        store = results_dictionary[k]
        extended_keys = [k] * len(store)
        df2[key] = extended_keys
        df2[value] = store
        df = df.append(df2, ignore_index=True, sort=False)
    return df


def fill_results_from_graph(
    results_df,
    scaffold,
    fields=['neuron_number', 'name', 'brick'],
):
    field_values = {'name': deque()}
    if 'name' in fields:
        fields.remove('name')
    for node in scaffold.graph.nodes:
        field_values['name'].append(node)
        for field in fields:
            if field not in field_values:
                field_values[field] = deque()
            if field in scaffold.graph.nodes[node]:
                field_values[field].append(scaffold.graph.nodes[node][field])
            else:
                field_values[field].append(np.NaN)
    field_atlas = pd.DataFrame(field_values)
    return results_df.merge(field_atlas, how='left')


def get_spiked_info(result, scaffold):
    result = fill_results_from_graph(result,
                                     scaffold,
                                     fields=['time', 'neuron_number', 'name'])
    return list(result['name'])


def results_dict(result, scaffold):
    times = np.array(result['time'])
    max_time = int(np.max(times))
    return {
        t: list(result[result['time'] == t]['name'])
        for t in range(0, max_time + 1)
    }


def set_circuit_position(scaffold):
    degree = dict(scaffold.circuit.out_degree())
    pos = {}
    input_layer = []
    for node in scaffold.circuit.nodes():
        brick = scaffold.circuit.nodes[node]['name']
        for bricks in scaffold.circuit.nodes():
            if brick == scaffold.circuit.nodes[bricks]['name']:
                if 'layer' in scaffold.circuit.nodes[bricks]:
                    if scaffold.circuit.nodes[bricks]['layer'] == 'input':
                        input_layer.append(node)
    max_degree = max([degree[n] for n in scaffold.circuit.nodes()])

    i = 70 * max_degree * len(input_layer)
    for node in input_layer:
        pos[node] = (0, i)
        i = i - 70 * max_degree
    j_val = {}
    for edge in scaffold.circuit.edges():
        if edge[0] in pos.keys() and edge[1] not in pos.keys():

            x = pos[edge[0]][0]
            y = pos[edge[0]][1]
            edge0_d = degree[edge[0]]
            if edge0_d % 2 == 1 and edge[0] not in j_val.keys():
                j_val[edge[0]] = y + 70 * np.floor(edge0_d / 2)
            elif edge0_d % 2 == 0 and edge[0] not in j_val.keys():
                j_val[edge[0]] = y + 35 + 70 * (edge0_d / 2 - 1)
            pos[edge[1]] = x + 250, j_val[edge[0]]
            j_val[edge[0]] = j_val[edge[0]] - 70

    if len(scaffold.circuit.nodes()) > len(pos):
        print(
            "Error! Not all nodes in circuit assigned positions. Unassigned nodes:"
        )
        for n in scaffold.circuit.nodes():
            if n not in pos.keys():
                print(n)
    return pos


def set_position(scaffold):
    MAX_NODES = 20
    total_length = 40 * scaffold.graph.number_of_nodes()
    total_height = 20 * scaffold.graph.number_of_nodes()
    pos = {}
    sorted_nodes_by_brick = {}

    for node in scaffold.graph.nodes():
        brick = scaffold.graph.nodes[node]['brick']
        if brick in sorted_nodes_by_brick.keys():
            sorted_nodes_by_brick[brick].append(node)
        else:
            sorted_nodes_by_brick[brick] = [node]

    num_cols = {}
    total_cols = 0
    for brick in sorted_nodes_by_brick:
        cols = 1
        total_cols += 1
        height = len(sorted_nodes_by_brick[brick])
        while height > MAX_NODES:
            height = height - MAX_NODES
            cols += 1
            total_cols += 1
        if cols == 1:
            num_cols[brick] = (cols, height)
        else:
            num_cols[brick] = (cols, MAX_NODES)
    dx = total_length / total_cols
    x = 0
    for brick in sorted_nodes_by_brick:
        current_cols = num_cols[brick][0]
        max_height = num_cols[brick][1]
        dy = total_height / max_height
        k = 0
        for i in range(0, current_cols):
            y = total_height
            for j in range(0, max_height):
                if k < len(sorted_nodes_by_brick[brick]):
                    pos[sorted_nodes_by_brick[brick][k]] = (x, y)
                    k += 1
                    y -= dy
            x += dx
    if scaffold.graph.number_of_nodes() > len(pos):
        print("Error! Not all nodes assigned positions. Unassigned nodes:")
        for n in scaffold.graph.nodes():
            if n not in pos.keys():
                print(n)
    return pos


def generate_gexf(scaffold, filename='fugu.gexf', result=None):
    """ Exports a scaffold to gexf.

    Exports a scaffold to a file using Graph Exchange XML Format (GEXF).

    This is largely a simple wrapper around NetworkX's `networkx.write_gexf`
    with the addition that neuron states can be embedded into the graph/node
    properties.


    """
    if not scaffold.is_built:
        raise ValueError("Scaffold should be built before exporting to gexf.")
    result = fill_results_from_graph(result,
                                     scaffold,
                                     fields=['time', 'neuron_number', 'name'])
    t_dict = results_dict(result, scaffold)
    graph = scaffold.graph
    max_t = int(np.max(np.array(result['time'])))
    G = nx.DiGraph(mode='dynamic')
    G.update(graph)
    for n in G.nodes():
        G.nodes[n]['spiked'] = []
        if 'index' in G.nodes[n]:
            G.nodes[n]['index'] = str(G.nodes[n]['index'])
    for node in G.nodes():
        for t in range(0, max_t + 1):
            if t in t_dict.keys():
                if node in t_dict[t]:
                    G.nodes[node]['spiked'].append(('spiked', t, t + 1))
                else:
                    G.nodes[node]['spiked'].append(('did not spike', t, t + 1))
            else:
                G.nodes[node]['spiked'].append(('did not spike', t, t + 1))

    nx.write_gexf(G, filename)
    return
