#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 15:31:37 2019

@author: lreeder
"""

import numpy as np
from utils import fill_results_from_graph
# Plotting imports
import dash
import dash_html_components as html
import dash_core_components as dcc
import dash_cytoscape as cyto
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import matplotlib.pyplot as plt
from collections import deque


###############################################################################
#
# General graph visualization for Fugu
# parameters: graph.............NetworkX DiGraph()
#             pos...............dict(node: (x,y))
#             result............Pandas DataFrame()
#             scaffold..........Fugu Scaffold()
#             view_scaffold.....boolean
#
# when running, it will output html link where vis can be accessed
# 
###############################################################################
def graph_vis(graph, pos, result, scaffold, view_scaffold=False):
    result = fill_results_from_graph(result,scaffold)
    result_dict = results_dict(result, scaffold)
    spiked_at_time = result_dict[0]
    
    nodes, edges = _build_node_and_edge_lists(graph, pos, spiked_at_time, spiked_color='pink', not_spiked_color='cadetblue')

        
    elements_ls = nodes + edges
    
    stylesheet_list = []
    stylesheet_list.append({'selector': 'node',
                            'style': {
                                    'content': 'data(id)',
                                    'width': '50',
                                    'height': '50'}})
    stylesheet_list.append({'selector': 'edge',
                            'style': {
                                    'curve-style': 'bezier',
                                    'target-arrow-shape': 'triangle',
                                    'label': 'data(weight)',
                                    'width': '7'}})
    stylesheet_list.append({'selector': '.pink',
                            'style': {
                                    'background-color': 'pink',
                                    'line-color': 'pink'}})
    stylesheet_list.append({'selector': '.cadetblue',
                            'style': {
                                    'background-color': 'cadetblue',
                                    'line-color': 'cadetblue'}})
    
    app = dash.Dash(__name__)
    
    max_t = 0  
    for t in result['time']:
        if t > max_t:
            max_t = int(t)
    time_step = {}
    for i in range(0,max_t+1):
        time_step[i] = i

    if not view_scaffold:
        app.layout = html.Div([
                cyto.Cytoscape(
                        id = 'graph',
                        elements=elements_ls,
                        layout={'name':'preset'},
                        autoRefreshLayout = False,
                        stylesheet=stylesheet_list),
                dcc.Textarea(
                        id = 'text-area',
                        draggable = True,
                        placeholder = 'Hover over a node to view its attributes',
                        readOnly = True,
                        value = '',
                        style={'width': '40%'}),
                html.P([
                        html.Label("Timestep"),
                        dcc.Slider(id='slider',
                                   marks = time_step,
                                   min = 0,
                                   max = max_t,
                                   value = 0,
                                   step = 1,
                                   updatemode='drag',
                                   )
                        ])
                ])
    else:
        scaffold_pos = set_position(scaffold.circuit, result)
        edge_trace = go.Scatter(x = [], y = [],
                                line = dict(width=12, color='#888'),
                                hoverinfo = 'none',
                                mode='lines')
        for edge in scaffold.circuit.edges():
            x0, y0 = scaffold_pos[edge[0]]
            x1, y1 = scaffold_pos[edge[1]]
            edge_trace['x'] += tuple([x0, x1, None])
            edge_trace['y'] += tuple([y0, y1, None])
        
        node_trace = go.Scatter(x = [], y = [], text = [],
                                mode = 'markers', hoverinfo = 'text',
                                marker = dict(color = [],
                                              size = 75,
                                              line = dict(width=2),
                                              symbol = 'hexagon'))
        for node in scaffold.circuit.nodes():
            x,y = scaffold_pos[node]
            node_trace['x'] += tuple([x])
            node_trace['y'] += tuple([y])
            node_trace['marker']['color'] += tuple(['darkseagreen'])
            node_info=str(scaffold.circuit.nodes[node]['name'])
            brick = '<br>Brick: ' + str(scaffold.circuit.nodes[node]['brick'])
            if 'layer' in scaffold.circuit.nodes[node]:    
                layer = '<br>Layer: ' + str(scaffold.circuit.nodes[node]['layer'])
                node_info = node_info + brick + layer
            else:
                node_info = node_info + brick
            node_trace['text'] += tuple([node_info])
        
        
        fig = go.Figure(data = [edge_trace,node_trace],
                        layout = go.Layout(
                                showlegend = False,
                                hovermode = 'closest',
                                hoverlabel = dict(bgcolor = 'aliceblue',
                                                bordercolor = 'gainsboro',
                                                font=dict(color = '#000000')),
                                xaxis = dict(
                                        showgrid = False,
                                        zeroline = False,
                                        showticklabels = False),
                                yaxis = dict(
                                        showgrid = False,
                                        zeroline = False,
                                        showticklabels = False)
                                ))
        app.layout = html.Div([
                html.Div(dcc.Graph(id = 'scaffold_graph', figure = fig)),
                cyto.Cytoscape(
                        id = 'graph',
                        elements = elements_ls,
                        layout = {'name':'preset'},
                        autoRefreshLayout = False,
                        stylesheet = stylesheet_list),
                dcc.Textarea(
                        id = 'text-area',
                        draggable = True,
                        placeholder = 'Hover over a node to view its attributes',
                        readOnly = True,
                        value = '',
                        style={'width': '40%'}),
                html.P([
                        html.Label("Timestep"),
                        dcc.Slider(id = 'slider',
                                   marks = time_step,
                                   min = 0,
                                   max = max_t,
                                   value = 0,
                                   step = 1,
                                   update_mode = 'drag')
                        ])
                        
                ])
        
    @app.callback([Output('text-area', 'value'),
                   Output('graph','elements')],
                  [Input('graph', 'mouseoverNodeData'),
                   Input('slider', 'value')])     
        
    def update_elements(node_data, X):
        elements_ls = []
        result_dict = results_dict(result,scaffold)
        spiked_at_time = result_dict[X]
        
        nodes, edges = _build_node_and_edge_lists(graph, pos, spiked_at_time)
        elements_ls = nodes + edges
        
        if node_data is not None:
            label = str(node_data['label'])
            num = '\nneuron number:  ' + str(node_data['id'])
            thresh = '\nthreshold:  ' + str(node_data['threshold'])
            value = label + num + thresh
            return value, elements_ls
        else:
            return '', elements_ls
    app.run_server(debug=True)
    return

###############################################################################
#
# Raster plot for Fugu
# parameters: graph.............NetworkX DiGraph()
#             result............Pandas DataFrame()
#             scaffold..........Fugu Scaffold()
# 
###############################################################################
def raster_plot(result, scaffold):
    result = fill_results_from_graph(result, scaffold)
    
    print(result)
    times = []
    nodes = []
    i = 0
    for t in result['time']:
        times.append(int(t))
        nodes.append(int(result['neuron_number'][i]))
        i = i + 1
        
    plt.figure(figsize=(20,10))
    plt.axis([0,times[-1]+1, 0, len(scaffold.graph.nodes())])
    plt.plot(times,nodes,'.')
    plt.xlabel("Simulation time")
    plt.ylabel("Neuron index")
    plt.title("Raster Plot")
    
    plt.show()
    return

def _build_node_and_edge_lists(graph, pos, spiked_at_time, spiked_color='pink', not_spiked_color='cadetblue'):
    nodes = deque()
    edges = deque()
    
    for node in graph.nodes:
        node_data = {}
        node_position = {'x': pos[node][0], 'y':pos[node][1]}
        node_data['id'] = graph.nodes[node]['neuron_number']
        node_data['label'] = node
        node_data['threshold'] = graph.nodes[node]['threshold']
        node_dict = {'data': node_data, 
                     'position': node_position, 
                     'classes': 'pink' if str(node) in spiked_at_time else 'cadetblue'}
        nodes.append(node_dict)
    
    for edge in graph.edges: 
    
        source = edge[0]
        target = edge[1]
           
        edges.append({'data': 
            {'source': graph.nodes[source]['neuron_number'],
                  'target': graph.nodes[target]['neuron_number'],
                  'label': str(source) + ' to ' + str(target),
                  'weight': graph.edges[edge]['weight'],
                  'delay': graph.edges[edge]['delay']}  
    })
    return list(nodes), list(edges)


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

if __name__ == "__main__":
    from fugu import Scaffold, Vector_Input, Shortest_Path_Length
    import networkx as nx

    skip_baseline = False
    scaffold = Scaffold()
    scaffold.add_brick(Vector_Input(np.array([1]), coding='Raster', name='Input0'), 'input' )
    print("Building Graph")
    target_graph = nx.generators.path_graph(5000)
    for edge in target_graph.edges:
        target_graph.edges[edge]['weight'] = 1.0
    scaffold.add_brick(Shortest_Path_Length(target_graph,0))
    
    print("Laying Bricks")
    scaffold.lay_bricks()
    scaffold.summary()
    print()
    print('----------------------')
    print('Results:', end='\n\n')
    print("Evaluating")
    result = scaffold.evaluate(backend='ds', max_runtime=10, record_all=True)
    print("Done evaluating.")
    print("Building Layout")
    pos = set_position(scaffold.graph)
    
    #use a scale for Networkx's layouts:
    scale = 10*nx.number_of_nodes(scaffold.graph)
    
    #following works best for small graphs:
    #input_layer = []
    #for node in scaffold.graph.nodes:
    #    input_layer.append(node)
    #for edge in scaffold.graph.edges:
    #    if edge[1] in input_layer:
    #        input_layer.remove(edge[1])
    #pos = nx.layout.spring_layout(scaffold.graph, scale=scale, pos=pos, fixed=input_layer)
    
    #Large graphs:
    pos = nx.layout.spring_layout(scaffold.graph, scale=scale, pos=pos)
    
    print("Rendering graph")
    graph_vis(scaffold.graph, pos, result, scaffold)