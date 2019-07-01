import numpy as np
from export_utils import fill_results_from_graph, set_position, results_dict
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
def graph_vis(graph, pos, result, scaffold):
    result = fill_results_from_graph(result,scaffold)
    result_dict = results_dict(result, scaffold)
    if 0 in result_dict.keys():
        spiked_at_time = result_dict[0]
    else:
        spiked_at_time = []
    
    nodes, edges = _build_node_and_edge_lists(graph, pos, spiked_at_time)

        
    elements_ls = nodes + edges
    
    stylesheet_list = []
    stylesheet_list.append({'selector': 'node',
                            'style': {
                                    #'content': 'data(id)',
                                    'width': '50',
                                    'height': '50'}})
    stylesheet_list.append({'selector': 'edge',
                            'style': {
                                    'curve-style': 'bezier',
                                    'target-arrow-shape': 'triangle',
                                    'label': 'data(weight)',
                                    'width': '7',
                                    'line-color': '#888'}})
    stylesheet_list.append({'selector': '.pink',
                            'style': {
                                    'background-color': '#DE5F76',
                                    'line-color': '#DE5F76'}})
    stylesheet_list.append({'selector': '.cadetblue',
                            'style': {
                                    'background-color': '#4B8F8C',
                                    'line-color': '#4B8F8C'}})
    max_t = 0  
    for t in result['time']:
        if t > max_t:
            max_t = int(t)
    time_step = {}
    for i in range(0,max_t+1):
        time_step[str(i)] = str(i)
    
    scaffold_pos = set_position(scaffold.circuit)
    edge_trace = go.Scatter(x = [], y = [],
                            line = dict(width=12, color='#888'),
                            hoverinfo = 'none',
                            mode = 'lines')
    for edge in scaffold.circuit.edges():
        x0, y0 = scaffold_pos[edge[0]]
        x1, y1 = scaffold_pos[edge[1]]
        edge_trace['x'] += tuple([x0, x1, None])
        edge_trace['y'] += tuple([y0, y1, None])
    
    node_trace = go.Scatter(x = [], y = [], text = [],
                            mode = 'markers', hoverinfo = 'text',
                            marker = dict(color = [],
                                          size = 75,
                                          line = dict(width=0),
                                          symbol = 'hexagon'))
    for node in scaffold.circuit.nodes():
        x,y = scaffold_pos[node]
        node_trace['x'] += tuple([x])
        node_trace['y'] += tuple([y])
        node_info=str(scaffold.circuit.nodes[node]['name'])
        brick = '<br>brick: ' + str(scaffold.circuit.nodes[node]['brick'])
        if 'layer' in scaffold.circuit.nodes[node]:    
            layer = '<br>layer: ' + str(scaffold.circuit.nodes[node]['layer'])
            node_info = node_info + brick + layer
            if scaffold.circuit.nodes[node]['layer'] == 'input':
                node_trace['marker']['color'] += tuple(['#DE5F76'])
            else:
                node_trace['marker']['color'] += tuple(['#4B8F8C'])
        else:
            node_info = node_info + brick
            node_trace['marker']['color'] += tuple(['#2C365E'])
        node_trace['text'] += tuple([node_info])
    
    
    fig = go.Figure(data = [edge_trace,node_trace],
                    layout = go.Layout(
                            font = dict(family="Arial", size=20, color="#444"),
                            title= dict(text="scaffold.circuit", x=0, y=0.98),
                            height = 400,
                            margin= dict(l=50,r=50,t=40,b=50),
                            paper_bgcolor = '#F5F5F5',
                            plot_bgcolor = '#F5F5F5',
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
                                    showticklabels = False),
                            ))
    spike_trace = go.Scatter(x = [], y = [], text = [],
                             mode='markers', 
                             hoverinfo='text',
                             textposition = 'top center',
                             marker= dict(color = '#2C365E',
                                          size = 5,
                                          line=dict(width=0))
                            )
    for time in result_dict:
        for node in result_dict[time]:
            x = time
            y = graph.nodes[node]['neuron_number']
            spike_trace['x'] += tuple([x])
            spike_trace['y'] += tuple([y])
            spike_trace['text'] += tuple([str(graph.nodes[node]['neuron_number']) + ': ' + str(node)])
    
    sub_amt = '0.'
    for i in range(0, len(str(nx.number_of_nodes(graph)))-1):
        sub_amt = sub_amt + '0'
    sub_amt = sub_amt + '2'
    sub_amt = float(sub_amt)
    raster_fig = go.Figure(data = [spike_trace],
                          layout = go.Layout(
                              font = dict(family="Arial", size=20, color="#444"),
                              title= dict(text="spike raster", x=0, y=1-sub_amt),
                              height=400 + 10*nx.number_of_nodes(graph),
                              margin=dict(l=25*len(str(nx.number_of_nodes(graph))), r=50, t=40, b=50),
                              paper_bgcolor = '#F5F5F5',
                              plot_bgcolor = '#F5F5F5',
                              showlegend=False,
                              hovermode='closest',
                              hoverlabel=dict(bgcolor='aliceblue',
                                             bordercolor='gainsboro',
                                             font=dict(color='#000000')),
                              xaxis = dict(
                                  title='time',
                                  fixedrange=True,
                                  range=[-0.05,max_t + 0.5],
                              ),
                              yaxis = dict(
                                  title='neuron number',
                                  fixedrange=True,
                                  range=[-0.5, nx.number_of_nodes(graph) + 1],
                                  nticks=nx.number_of_nodes(graph),
                              )
                            
                          ))
    external_stylesheets =['https://codepen.io/chriddyp/pen/bWLwgP.css']
    app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
    app.css.config.serve_locally = True
    app.scripts.config.serve_locally = True
    app.layout = html.Div(className="row",
                          children = [ 
                                html.Div(className="twelve columns",
                                         children=[
                                                 html.Label('fugu visualizations',
                                                            style={'font-family': 'Arial', 
                                                                   'font-size':'3.5rem', 
                                                                   'margin':'1px'}
                                                            )
                                                 ]
                                         ),
                                html.Div(className="twelve columns", children=[
                                    html.Div(className="six columns", 
                                             children = [dcc.Graph(id = 'scaffold_graph', 
                                                                   figure = fig),
                                                         html.Div(className="twelve columns", 
                                                                  children=[
                                                                          html.Label('scaffold.graph', 
                                                                                     style={'font-family': 'Arial', 
                                                                                            'font-size': '2.8rem', 
                                                                                            'background': '#F5F5F5', 
                                                                                            'margin-top':'15pt'}
                                                                                     )
                                                                          ]
                                                                  ),
                                                         cyto.Cytoscape(
                                                                id = 'graph',
                                                                #title='graph',
                                                                elements = elements_ls,
                                                                style = {'width': '100%',
                                                                         'height': '500px',
                                                                         'background': '#F5F5F5',
                                                                         'margin-top': '45pt'},
                                                                layout = {'name':'preset'},
                                                                autoRefreshLayout = False,
                                                                stylesheet = stylesheet_list),
                                                         dcc.Textarea(
                                                                id = 'text-area',
                                                                draggable = True,
                                                                placeholder = 'hover over a node to view its attributes',
                                                                readOnly = True,
                                                                value = '',
                                                                style={'width': '100%',
                                                                       'background': '#F5F5F5'}),
                                                        html.P(children=[
                                                                    html.Div(className="twelve", 
                                                                             children=[
                                                                                     html.Label("timestep", 
                                                                                                style={'font-family': 'Arial', 
                                                                                                       'font-size': '2.0rem', 
                                                                                                       'background': '#F5F5F5'}
                                                                                                )
                                                                                      ]
                                                                            ),
                                                                    html.Div(className="twelve columns", children=[
                                                                               dcc.Slider(id = 'slider',
                                                                               marks = time_step,
                                                                               min = 0,
                                                                               max = max_t,
                                                                               value = 0,
                                                                               step = 1,
                                                                               updatemode='drag',
                                                                              )]
                                                                            ),                                                                    
                                                                ])
                                                        ]),
                                                                             
                                    html.Div(className="six columns",
                                             children = [html.Div(className="twelve columns", children = [
                                                     
                                                                 html.Label('brick info', style={'font-family': 'Arial', 
                                                                                                 'font-size': '2.8rem',
                                                                                                 'background': '#F5F5F5'}),
                                                                dcc.Textarea(
                                                                        id = 'summary',
                                                                        draggable=True,
                                                                        placeholder = 'spiking node info...' + '\nbrick summary info...',
                                                                        readOnly = True,
                                                                        value = '',
                                                                        style={'width':'100%',
                                                                               'background':'#F5F5F5',
                                                                               'margin-bottom': '15pt'}),
                                                                dcc.Graph(id='raster_plot', figure=raster_fig)]),
                                                         ]),
                                ])
                            ])
    
    @app.callback([Output('text-area', 'value'),
                   Output('graph','elements')],
                  [Input('graph', 'mouseoverNodeData'),
                   Input('slider', 'value')])     
    
    def update_elements(node_data, X):
        elements_ls = []
        result_dict = results_dict(result,scaffold)
        if X in result_dict.keys():
            spiked_at_time = result_dict[X]
        else:
            spiked_at_time = []
        nodes, edges = _build_node_and_edge_lists(graph, pos, spiked_at_time)
        elements_ls = nodes + edges
        
        if node_data is not None:
            label = str(node_data['label'])
            num = '\nneuron number:  ' + str(node_data['id'])
            thresh = '\nthreshold:  ' + str(node_data['threshold']) + '\n'
            value = label + num + thresh
            return value, elements_ls
        else:
            return '', elements_ls
        
#    app.css.append_css({
#        'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'
#    })
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






if __name__ == "__main__":
    from fugu.scaffold import Scaffold
    from fugu.bricks import Vector_Input, Copy, Dot, Threshold, Shortest_Path
    import networkx as nx
    scaffold = Scaffold()
    print("Building Graph")
    #Small graph:
    scaffold.add_brick(Vector_Input(np.array([1,0,1]), coding='Raster'), input_nodes='input')
    scaffold.add_brick(Copy())
    scaffold.add_brick(Dot([1,0,1], name='ADotOperator'), (1,0))
    scaffold.add_brick(Dot([0,0,1], name='AnotherDotOperator'), (1,1))
    scaffold.add_brick(Threshold(1.75, name='Neuron13'), (2,0), output=True)
    scaffold.add_brick(Threshold(1.25, name='Neuron14'), (3,0), output=True)
    #Large graph:
    #scaffold.add_brick(Vector_Input(np.array([1]), coding='Raster', name='Input0'), 'input' )
    #target_graph = nx.generators.path_graph(5000)
    #for edge in target_graph.edges:
    #    target_graph.edges[edge]['weight'] = 1.0
    #scaffold.add_brick(Shortest_Path(target_graph,0))
    print("Laying Bricks")
    scaffold.lay_bricks()
    scaffold.summary()
    print()
    print('----------------------')
    print ('Results:', end='\n\n')
    print("Evaluating")
    result = scaffold.evaluate(backend='ds', max_runtime=10, record_all=True)
    print("Done evaluating.")
    print("Building Layout")
    pos = set_position(scaffold.graph)
    
    #use a scale for Networkx's layouts:
    #scale = 10*nx.number_of_nodes(scaffold.graph)
    
    #following works best for small graphs:
    #input_layer = []
    #for node in scaffold.graph.nodes:
    #    input_layer.append(node)
    #for edge in scaffold.graph.edges:
    #    if edge[1] in input_layer:
    #        input_layer.remove(edge[1])
    #pos = nx.layout.spring_layout(scaffold.graph, pos=pos, fixed=input_layer)
    
    #Large graphs:
    #
    #pos = nx.layout.spring_layout(scaffold.graph, scale=scale, pos=pos)
    
    print("Rendering graph")
    graph_vis(scaffold.graph, pos, result, scaffold)
