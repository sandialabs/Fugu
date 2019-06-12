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


###############################################################################
#
# General graph visualization for Fugu
# parameters: graph.............NetworkX DiGraph()
#             result............Pandas DataFrame()
#             scaffold..........Fugu Scaffold()
#             view_scaffold.....boolean
#
# when running, it will output html link where vis can be accessed
# 
###############################################################################
def graph_vis(graph, result, scaffold, view_scaffold=False):
    spiked = get_spiked_info(graph, result, scaffold)
    pos = set_position(graph, result)
    
    nodes = []
    edges = []
    for edge in graph.edges():
        source_dict = {}
        source_data = {}
        source_position = {}
        
        target_dict = {}
        target_data = {}
        target_position = {}
        
        e = {}
        e_data = {}
        
        source = edge[0]
        target = edge[1]
        
        source_position = {'x': pos[source][0], 'y':pos[source][1]}
        source_data['id'] = graph.nodes[source]['neuron_number']
        source_data['label'] = source
        source_data['threshold'] = graph.nodes[source]['threshold']
        source_dict = {'data': source_data, 'position': source_position}
        
        target_position = {'x': pos[target][0], 'y': pos[target][1]}
        target_data['id'] = graph.nodes[target]['neuron_number']
        target_data['label'] = target
        target_data['threshold'] = graph.nodes[target]['threshold']
        target_dict = {'data': target_data, 'position': target_position}
        
        if str(source) in spiked and str(target) in spiked:
            source_dict['classes'] = 'pink'
            target_dict['classes'] = 'pink'
        elif str(source) in spiked and str(target) not in spiked:
            source_dict['classes'] = 'pink'
            target_dict['classes'] = 'cadetblue'
        elif str(source) not in spiked and str(target) in spiked:
            source_dict['classes'] = 'cadetblue'
            target_dict['classes'] = 'pink'
        else:
            source_dict['classes'] = 'cadetblue'
            target_dict['classes'] = 'cadetblue'
        
        prev_source = False
        prev_target = False
        for other_dict in nodes:
            if source_dict == other_dict:
                prev_source = True
            if target_dict == other_dict:
                prev_target = True
        if prev_source == False:
            nodes.append(source_dict)
        if prev_target == False:
            nodes.append(target_dict)
        
        e_data = {'source': graph.nodes[source]['neuron_number'],
                  'target': graph.nodes[target]['neuron_number'],
                  'label': str(source) + ' to ' + str(target),
                  'weight': graph.edges[edge]['weight'],
                  'delay': graph.edges[edge]['delay']}
        e['data'] = e_data
        edges.append(e)
        
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
                                    'background-color': 'cadetblue',
                                    'line-color': 'cadetblue'}})
    stylesheet_list.append({'selector': '.cadetblue',
                            'style': {
                                    'background-color': 'cadetblue',
                                    'line-color': 'cadetblue'}})
    
    result = fill_results_from_graph(result, scaffold)
    i = 0
    max_t = 0
    t_dict = {}
    for t in result['time']:
        if t > max_t:
            max_t = int(t)
        if t not in t_dict.keys():
            t_dict[int(t)] = [result['name'][i]]
        else:
            t_dict[int(t)].append(result['name'][i])
        i = i + 1
    time_step = {}
    for i in range(0,max_t+1):
        time_step[i] = i
  
    if view_scaffold:
        scaffold_pos = set_position(scaffold.circuit, result)
        edge_trace = go.Scatter(x = [], y = [],
                                line = dict(width=12, color='#888'),
                                            hoverinfo='none',
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
            x, y = scaffold_pos[node]
            node_trace['x'] += tuple([x])
            node_trace['y'] += tuple([y])
            node_trace['marker']['color'] += tuple(['darkseagreen'])
            if 'layer' in scaffold.circuit.nodes[node]:
                node_info=str(scaffold.circuit.nodes[node]['name'])
                brick='<br>Brick: '+str(scaffold.circuit.nodes[node]['brick'])
                layer='<br>Layer: '+str(scaffold.circuit.nodes[node]['layer'])
                node_info = node_info + brick + layer
            else:
                node_info = str(scaffold.circuit.nodes[node]['name'])
                brick='<br>Brick: '+str(scaffold.circuit.nodes[node]['brick'])
                node_info = node_info + brick
                
            node_trace['text'] += tuple([node_info])
            
        app = dash.Dash(__name__)
        fig = go.Figure(data=[edge_trace, node_trace],
                        layout = go.Layout(
                                showlegend = False,
                                hovermode = 'closest',
                                hoverlabel = dict(bgcolor='aliceblue',
                                                  bordercolor='gainsboro',
                                                  font = dict(color='#000000')),
                                xaxis = dict(
                                        showgrid=False,
                                        zeroline=False,
                                        showticklabels=False
                                        ),
                                yaxis = dict(
                                        showgrid=False,
                                        zeroline=False,
                                        showticklabels=False
                                        )))

        app.layout = html.Div([
                html.Div(dcc.Graph(id='scaffold_graph', figure=fig)),
                cyto.Cytoscape(
                        id = 'graph',
                        elements=elements_ls,
                        layout={'name':'preset'},
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
        
        
        @app.callback([Output('text-area', 'value'),
                       Output('graph','stylesheet')],
                      [Input('graph', 'mouseoverNodeData'),
                       Input('slider', 'value')])     
        
        def update_stylesheet(node_data, X):
            if X in t_dict.keys():
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
                                                'content': 'spiked',
                                                'background-color': 'pink',
                                                'line-color': 'pink'}})
                stylesheet_list.append({'selector': '.cadetblue',
                                        'style': {
                                                'background-color': 'cadetblue',
                                                'line-color': 'cadetblue'}})
                if node_data is not None:
                    label = str(node_data['label'])
                    num = '\nneuron number:  ' + str(node_data['id'])
                    thresh = '\nthreshold:  ' + str(node_data['threshold'])
                    value = label + num + thresh
                    return value, stylesheet_list
                else:
                    return '', stylesheet_list
            else:
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
                                                'background-color': 'cadetblue',
                                                'line-color': 'cadetblue'}})
                stylesheet_list.append({'selector': '.cadetblue',
                                        'style': {
                                                'background-color': 'cadetblue',
                                                'line-color': 'cadetblue'}})
               
                if node_data is not None:
                    label = str(node_data['label'])
                    num = '\nneuron number:  ' + str(node_data['id'])
                    thresh = '\nthreshold:  ' + str(node_data['threshold'])
                    value = label + num + thresh
                    return value, stylesheet_list
                else:
                    return '', stylesheet_list
        
        app.run_server(debug=True)
        return
    
    else :
        app = dash.Dash(__name__)
        app.layout = html.Div([
                cyto.Cytoscape(
                        id = 'graph',
                        elements=elements_ls,
                        layout={'name':'preset'},
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
        
        
        @app.callback([Output('text-area', 'value'),
                       Output('graph','stylesheet')],
                      [Input('graph', 'mouseoverNodeData'),
                       Input('slider', 'value')])     
        
        def update_stylesheet(node_data, X):
            if X in t_dict.keys():
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
                                                'content': 'spiked',
                                                'background-color': 'pink',
                                                'line-color': 'pink'}})
                stylesheet_list.append({'selector': '.cadetblue',
                                        'style': {
                                                'background-color': 'cadetblue',
                                                'line-color': 'cadetblue'}})
                
                if node_data is not None:
                    label = str(node_data['label'])
                    num = '\nneuron number:  ' + str(node_data['id'])
                    thresh = '\nthreshold:  ' + str(node_data['threshold'])
                    value = label + num + thresh
                    return value, stylesheet_list
                else:
                    return '', stylesheet_list
            else:
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
                                                'background-color': 'cadetblue',
                                                'line-color': 'cadetblue'}})
                stylesheet_list.append({'selector': '.cadetblue',
                                        'style': {
                                                'background-color': 'cadetblue',
                                                'line-color': 'cadetblue'}})
                
                
                
                if node_data is not None:
                    label = str(node_data['label'])
                    num = '\nneuron number:  ' + str(node_data['id'])
                    thresh = '\nthreshold:  ' + str(node_data['threshold'])
                    value = label + num + thresh
                    return value, stylesheet_list
                else:
                    return '', stylesheet_list
        
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
def raster_plot(graph, result, scaffold):
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



def get_spiked_info(graph, result, scaffold):
    spiked = []
    result = fill_results_from_graph(result, scaffold, fields=['time', 'neuron_number', 'name'])
    s = result['name']
    for i in range(0,len(s)):
        spiked.append(s[i])
    return spiked

def set_position(graph, result):
    
    degree = {}
    for edge in graph.edges():
        d = 0
        for e in graph.edges():
            if edge[0] == e[0]:
                d = d + 1
        degree[edge[0]] = d
    for n in graph.nodes():
        if n not in degree.keys():
            degree[n] = 0

    incoming = {}
    for edge in graph.edges():
        i = 0
        for e in graph.edges():
            if edge[1] == e[1]:
                i = i + 1
        incoming[edge[1]] = i
    for n in graph.nodes():
        if n not in incoming.keys():
            incoming[n] = 0
    
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
#            if incoming[edge[1]] > 1:
#                previous = {}
#                #store previous edges and y values that connect to this
#                for e in graph.edges():
#                    if e[1] == edge[1]:
#                        previous[edge[0]] = pos[edge[0]][1]
#                x = pos[edge[0]][0]
#                avg = 0
#                k = 0
#                for key, height in previous.items():
#                    avg = avg + height
#                    k = k + 1
#                 
#                pos[edge[1]] = x + 250, avg/k
                
#            else:
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

    