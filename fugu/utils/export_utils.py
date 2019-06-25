import networkx as nx
import pandas as pd
import numpy as np
from collections import deque

def results_df_from_dict(results_dictionary, key, value):
    df = pd.DataFrame()
    for k in results_dictionary:
        df2 = pd.DataFrame()
        store = results_dictionary[k]
        extended_keys = [k]*len(store)
        df2[key] = extended_keys
        df2[value] = store
        df = df.append(df2, ignore_index=True, sort=False)
#        df = df.astype({value: int})
    return df

def fill_results_from_graph(results_df, scaffold, fields = ['time', 'neuron_number', 'name', 'brick'], unmatched_fields=['time']):
    field_values = {'name':deque()}
    if 'name' in fields:
        fields.remove('name')
    for node in scaffold.graph.nodes:
        field_values['name'].append(node)
        for field in [field for field in fields if field not in unmatched_fields ]:
            if field not in field_values:
                field_values[field] = deque()
            if field in scaffold.graph.nodes[node]:
                field_values[field].append(scaffold.graph.nodes[node][field])
            else:
                field_values[field].append(np.NaN)
    field_atlas = pd.DataFrame(field_values)
    return results_df.merge(field_atlas, how='left')

def get_spiked_info(result, scaffold):
    result = fill_results_from_graph(result, scaffold, fields=['time', 'neuron_number', 'name'])
    return list(result['name'])

def results_dict(result, scaffold):
    result = fill_results_from_graph(result, scaffold, fields=['time', 'neuron_number', 'name'])
    i = 0
    t_dict = {}
    for t in result['time']:
        if int(t) not in t_dict.keys():
            t_dict[int(t)] = [result['name'][i]]
        else:
            t_dict[int(t)].append(result['name'][i])
        i = i + 1
    return(t_dict)

def set_position(graph):
    degree = dict(graph.out_degree())

    pos = {}
    input_layer = []
    output_layer = []
    middle_layer = []
    for node in graph.nodes():
        input_layer.append(node)
        output_layer.append(node)
        middle_layer.append(node)
    for edge in graph.edges():
        if edge[1] in input_layer:
            input_layer.remove(edge[1])
        if edge[0] in output_layer:
            output_layer.remove(edge[0])
    for node in input_layer:
        if node in middle_layer:
            middle_layer.remove(node)
    for node in output_layer:
        if node in middle_layer:
            middle_layer.remove(node)
    
    max_degree = 0
    for n in input_layer:
        if degree[n] > max_degree:
            max_degree = degree[n]
    
    i = 70*max_degree*len(input_layer)
    for node in input_layer:
        pos[node] = (0, i)
        i = i - 70*max_degree
    j_val = {}
    for edge in graph.edges():
        if edge[0] in pos.keys() and edge[1] not in pos.keys():

            x = pos[edge[0]][0]
            y = pos[edge[0]][1]
            edge0_d = degree[edge[0]]
            if edge0_d%2 == 1 and edge[0] not in j_val.keys():
                j_val[edge[0]] = y + 70*np.floor(edge0_d/2)
            elif edge0_d%2 == 0 and edge[0] not in j_val.keys():
                j_val[edge[0]] = y + 35 + 70*(edge0_d/2 - 1)
            pos[edge[1]] = x + 250, j_val[edge[0]]
            j_val[edge[0]] = j_val[edge[0]] - 70
                
            
    if len(graph.nodes()) < len(pos):
        print("Error! Not all nodes assigned positions. Unassigned nodes:")
        for n in graph.nodes():
            if n not in pos.keys():
                print(n)
    return pos

def generate_gexf(graph, result, scaffold, filename='fugu.gexf'):
    result = fill_results_from_graph(result, scaffold, fields=['time', 'neuron_number', 'name'])
    pos = set_position(graph)
    
    max_t = 0
    for t in result['time']:
        if t > max_t:
            max_t = int(t)
    
    t_dict = results_dict(result, scaffold)
    G = nx.DiGraph(mode='dynamic')
    for n in graph.nodes():
        node = graph.nodes[n]['neuron_number']
        if 'index' in graph.nodes[n].keys():
            G.add_node(node, spiked = [], index = str(graph.nodes[n]['index']), 
                       threshold=graph.nodes[n]['threshold'], decay=graph.nodes[n]['decay'], 
                       brick=graph.nodes[n]['brick'], name = n, 
                       x = pos[n][0], y = int(pos[n][1]))
        else:
            G.add_node(node, spiked = [], threshold=graph.nodes[n]['threshold'],
                       decay=graph.nodes[n]['decay'], brick=graph.nodes[n]['brick'], 
                       name=n, x=pos[n][0], y=int(pos[n][1]))
    
    for t in range(0, max_t+1):
        spiked_at_t = t_dict[t]
        for node in G.nodes():
            if G.nodes[node]['name'] in spiked_at_t:
                G.nodes[node]['spiked'].append(('spiked', t, t+1))
            else:
                G.nodes[node]['spiked'].append(('', t, t+1))
    
    for edge in graph.edges():
        e0 = graph.nodes[edge[0]]['neuron_number']
        e1 = graph.nodes[edge[1]]['neuron_number']
        e_data = graph.edges[edge]
        G.add_edge(e0, e1, data=e_data)
    
    nx.write_gexf(G, filename)
    
    print("Maximum spiked time: ", max_t+1)
    return
