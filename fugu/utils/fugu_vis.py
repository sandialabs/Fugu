import numpy as np
from export_utils import fill_results_from_graph, set_position, set_circuit_position, results_dict
# Plotting imports
import dash
import dash_html_components as html
import dash_core_components as dcc
import dash_cytoscape as cyto
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import matplotlib.pyplot as plt
from collections import deque
import base64
import os


def graph_vis(result,
              scaffold,
              pos=None,
              debug=False,
              raster=True,
              check_spikes=None):
    """
    General graph visualization for Fugu. Creates server using Plotly's Dash.
    Will output a link to access html server.
    
    Arguments:
        + result : - pandas DataFrame 
            + stores result of fugu simulation
        + scaffold : - fugu Scaffold
            + stores scaffold object
        + pos : - dict, optional
            + dictionary of node positions (default is None)
        + debug : - bool, optional
            + toggle if dash server is on debugging mode (default is False)
        + raster : - bool, optional
            + toggle if raster plot is showed (default is True)
        + check_spikes : - dict, optional
            + dictionary specifying what spikes to check (default is None)
            + has format {neuron_number: {time_step: spike}} where spikes are denoted by 1 and no spike is 0
            + for example, to check if neuron 3 spiked at time 0 and didn't spike at time 2, {3: {0: 1, 2: 0}}
    """
    graph = scaffold.graph
    if pos == None:
        pos = set_position(scaffold)
    result = fill_results_from_graph(result, scaffold)
    result_dict = results_dict(result, scaffold)

    if 0 in result_dict.keys():
        spiked_at_time = result_dict[0]
    else:
        spiked_at_time = []

    nodes, edges = _build_node_and_edge_lists(graph, pos, spiked_at_time)
    elements_ls = nodes + edges

    stylesheet_list = []
    stylesheet_list.append({
        'selector': 'node',
        'style': {
            #'content': 'data(id)',
            'width': str(16 * np.ceil(graph.number_of_nodes() / 100)),
            'height': str(16 * np.ceil(graph.number_of_nodes() / 100))
        }
    })
    stylesheet_list.append({
        'selector': 'edge',
        'style': {
            'curve-style': 'bezier',
            'target-arrow-shape': 'triangle',
            #'label': 'data(weight)',
            'width': str(16 * np.ceil(graph.number_of_nodes() / 100) / 6),
            'line-color': '#A0A0A0'
        }
    })
    stylesheet_list.append({
        'selector': '.spikes',
        'style': {
            'background-color': '#7D0D7C',
            'line-color': '#7D0D7C'
        }
    })
    stylesheet_list.append({
        'selector': '.nonspikes',
        'style': {
            'background-color': '#008E74',
            'line-color': '#008E74'
        }
    })

    max_t = int(np.max(np.array(result['time'])))
    time_step = {str(i): str(i) for i in range(0, max_t + 1)}
    max_deg = max(
        [scaffold.circuit.out_degree(n) for n in scaffold.circuit.nodes()])
    scaffold_pos = set_circuit_position(scaffold)

    edge_trace = go.Scatter(x=[],
                            y=[],
                            line=dict(width=12, color='#A0A0A0'),
                            hoverinfo='none',
                            mode='lines')
    for edge in scaffold.circuit.edges():
        x0, y0 = scaffold_pos[edge[0]]
        x1, y1 = scaffold_pos[edge[1]]
        edge_trace['x'] += tuple([x0, x1, None])
        edge_trace['y'] += tuple([y0, y1, None])

    node_trace = go.Scatter(x=[],
                            y=[],
                            text=[],
                            mode='markers',
                            hoverinfo='text',
                            marker=dict(color=[],
                                        size=75,
                                        line=dict(width=0),
                                        symbol='hexagon'))

    for node in scaffold.circuit.nodes():
        x, y = scaffold_pos[node]
        node_trace['x'] += tuple([x])
        node_trace['y'] += tuple([y])
        node_info = str(scaffold.circuit.nodes[node]['name'])
        brick = '<br>brick: ' + str(scaffold.circuit.nodes[node]['brick'])
        if 'layer' in scaffold.circuit.nodes[node]:
            layer = '<br>layer: ' + str(scaffold.circuit.nodes[node]['layer'])
            node_info = node_info + brick + layer
            if scaffold.circuit.nodes[node]['layer'] == 'input':
                node_trace['marker']['color'] += tuple(
                    ['#A92C00'])  #input nodes #DE5F76
            else:
                node_trace['marker']['color'] += tuple(
                    ['#6CB312'])  #output nodes #4B8F8C
        else:
            node_info = node_info + brick
            node_trace['marker']['color'] += tuple(['#FFA033'
                                                    ])  #other nodes #2C365E
        node_trace['text'] += tuple([node_info])

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        font=dict(family="sans-serif", size=20, color="#444"),
                        title=dict(text="scaffold.circuit", x=0, y=0.98),
                        height=215 * max_deg - 80,
                        margin=dict(l=50, r=50, t=40, b=20),
                        paper_bgcolor='#FFFFFF',
                        plot_bgcolor='#FFFFFF',
                        showlegend=False,
                        hovermode='closest',
                        hoverlabel=dict(bgcolor='aliceblue',
                                        bordercolor='gainsboro',
                                        font=dict(color='#000000')),
                        xaxis=dict(showgrid=False,
                                   zeroline=False,
                                   showticklabels=False),
                        yaxis=dict(showgrid=False,
                                   zeroline=False,
                                   showticklabels=False),
                    ))

    if raster:
        spike_trace = go.Scatter(x=[],
                                 y=[],
                                 text=[],
                                 mode='markers',
                                 hoverinfo='text',
                                 textposition='top center',
                                 marker=dict(color='#005376',
                                             size=9,
                                             line=dict(width=0)))
        x_range = []
        y_range = []
        for time in result_dict:
            for node in result_dict[time]:
                x = time
                x_range.append(time)
                y = graph.nodes[node]['neuron_number']
                y_range.append(graph.nodes[node]['neuron_number'])
                spike_trace['x'] += tuple([x])
                spike_trace['y'] += tuple([y])
                spike_trace['text'] += tuple([
                    str(graph.nodes[node]['neuron_number']) + ': ' + str(node)
                ])

        sub_amt = '0.'
        for i in range(0, len(str(graph.number_of_nodes())) - 1):
            sub_amt = sub_amt + '0'
        sub_amt = sub_amt + '2'
        sub_amt = float(sub_amt)
        raster_fig = go.Figure(
            data=[spike_trace],
            layout=go.Layout(font=dict(family="sans-serif",
                                       size=20,
                                       color="#444"),
                             title=dict(text="spike raster",
                                        x=0,
                                        y=1 - sub_amt),
                             height=400 + 10 * len(y_range),
                             margin=dict(l=25 *
                                         len(str(graph.number_of_nodes())),
                                         r=50,
                                         t=40,
                                         b=50),
                             paper_bgcolor='#FFFFFF',
                             plot_bgcolor='#FFFFFF',
                             showlegend=False,
                             hovermode='closest',
                             hoverlabel=dict(bgcolor='aliceblue',
                                             bordercolor='gainsboro',
                                             font=dict(color='#000000')),
                             xaxis=dict(
                                 title='time',
                                 type='category',
                                 tickmode='array',
                                 tickvals=x_range,
                                 ticktext=x_range,
                                 showgrid=True,
                                 zeroline=True,
                                 gridcolor='#DCDCDC',
                                 zerolinecolor='#DCDCDC',
                             ),
                             yaxis=dict(
                                 title='neuron number',
                                 type='category',
                                 tickmode='array',
                                 tickvals=y_range,
                                 ticktext=y_range,
                                 showgrid=True,
                                 zeroline=True,
                                 gridcolor='#DCDCDC',
                                 zerolinecolor='#DCDCDC',
                             )))
    if check_spikes is not None:
        z = []
        text = []
        end_time = list(result_dict.keys())[-1]
        for node in check_spikes.keys():
            y = []
            t = []
            for time in range(0, end_time + 1):
                actual_nodes = result_dict[time]
                actual_nums = [
                    scaffold.graph.nodes[neuron]['neuron_number']
                    for neuron in actual_nodes
                ]
                if time in check_spikes[node].keys():
                    spiked = check_spikes[node][time]
                    if spiked == 1 and node in actual_nums:
                        y.append(0)
                        t.append('expected: spike<br>result: spike')
                    elif spiked == 1 and node not in actual_nums:
                        y.append(1)
                        t.append('expected: spike<br>result: no spike')
                    elif spiked == 0 and node in actual_nums:
                        y.append(1)
                        t.append('expected: no spike<br>result: spike')
                    elif spiked == 0 and node not in actual_nums:
                        y.append(0)
                        t.append('expected: no spike<br>result: no spike')
                    else:
                        y.append(1)
                        t.append('error: incorrect format given')
                else:
                    y.append(0.5)
                    t.append('expected spike results not available')
            z.append(y)
            text.append(t)
        red = False
        green = False
        grey = False
        for y_neurons in z:
            for x_times in y_neurons:
                if x_times == 1:
                    red = True
                elif x_times == 0:
                    green = True
                elif x_times == 0.5:
                    grey = True
        if red and green and grey:
            colorscale = [[0, "rgb(108, 179, 18)"], [0.5, "#DCDCDC"],
                          [1, "rgb(180,20,0)"]]
        elif red and green and grey == False:
            colorscale = [[0, "rgb(108, 179, 18)"], [1, "rgb(180,20,0)"]]
        elif red and green == False and grey == False:
            colorscale = [[0, "rgb(180,20,0)"], [1, "rgb(180,20,0)"]]
        elif red and green == False and grey:
            colorscale = [[0, "#DCDCDC"], [1, "rgb(180,20,0)"]]
        elif red == False and green and grey:
            colorscale = [[0, "rgb(108, 179, 18)"], [1, "#DCDCDC"]]
        elif red == False and green == False and grey:
            colorscale = [[0, "#DCDCDC"], [1, "#DCDCDC"]]
        elif red == False and green and grey == False:
            colorscale = [[0, "rgb(108, 179, 18)"], [1, "rgb(108, 179, 18)"]]

        expected_trace = go.Heatmap(z=z,
                                    y=list(check_spikes.keys()),
                                    x=list(range(0, end_time + 1)),
                                    text=text,
                                    colorscale=colorscale,
                                    showscale=False)
        expected_data = [expected_trace]
        expected_layout = go.Layout(
            font=dict(family="sans-serif", size=20, color="#444"),
            title=dict(text='expected spike results', x=0, y=0.98),
            xaxis=dict(type='category',
                       tickmode='array',
                       tickvals=list(range(0, end_time + 1)),
                       ticktext=list(range(0, end_time + 1)),
                       title='time vals'),
            yaxis=dict(type='category',
                       tickmode='array',
                       tickvals=list(check_spikes.keys()),
                       ticktext=list(check_spikes.keys()),
                       title='neuron number'),
            height=300,
            margin=dict(b=40))
        expected_fig = go.Figure(data=expected_data, layout=expected_layout)

    output_options = []
    for node in graph.nodes:
        for brick in scaffold.circuit.nodes:
            if 'layer' in scaffold.circuit.nodes[brick]:
                if 'output' == scaffold.circuit.nodes[brick]['layer']:
                    if graph.nodes[node]['brick'] == scaffold.circuit.nodes[
                            brick]['name']:
                        output_options.append({
                            'label': str(node),
                            'value': str(node)
                        })

    if check_spikes is None and raster:
        right_side = _build_right_div_raster(output_options, raster_fig)
    elif check_spikes is None and raster == False:
        right_side = _build_right_div(output_options)
    elif check_spikes is not None and raster:
        right_side = _build_right_div_raster_expected(output_options,
                                                      raster_fig, expected_fig)
    else:
        right_side = _build_right_div_expected(output_options, expected_fig)
    external_stylesheets = ['//assets/custom.css']
    cwd = str(os.getcwd())
    snl_filename = cwd + '/assets/SNL_Stacked_Black_Blue.png'
    doe_filename = cwd + '/assets/New_DOE_Logo_Color.png'
    nnsa_filename = cwd + '/assets/NNSA_Logo_Color.png'
    fugu_filename = cwd + '/assets/Fugu_Logo_v0.png'
    snl = base64.b64encode(open(snl_filename, 'rb').read())
    doe = base64.b64encode(open(doe_filename, 'rb').read())
    nnsa = base64.b64encode(open(nnsa_filename, 'rb').read())
    fugu = base64.b64encode(open(fugu_filename, 'rb').read())
    with open(cwd + '/assets/funding_statement.txt', 'r') as myfile:
        funding_statement = myfile.read()
    app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
    app.css.config.serve_locally = True
    app.scripts.config.serve_locally = True
    app.title = "fugu"
    app.layout = html.Div(
        className="row",
        children=[
            html.Div(className="twelve columns",
                     children=[
                         html.Img(src='data:iamge/png;base64,{}'.format(
                             fugu.decode()),
                                  style={
                                      'width': '7%',
                                      'display': 'inline'
                                  }),
                         html.Img(src='data:image/png;base64,{}'.format(
                             snl.decode()),
                                  style={
                                      'width': '10%',
                                      'display': 'inline',
                                      'position': 'absolute',
                                      'right': '0'
                                  })
                     ]),
            html.Div(children=[
                _build_left_div(fig, elements_ls, stylesheet_list, time_step,
                                max_t), right_side
            ]),
            html.Div(
                className="twelve columns",
                children=[
                    html.Div(
                        className="footer",
                        children=[
                            html.Div([
                                html.Img(src='data:image/png;base64,{}'.format(
                                    doe.decode()),
                                         style={
                                             'width': '45%',
                                             'margin-right': '0.5em',
                                             'display': 'inline'
                                         }),
                                html.Img(src='data:image/png;base64,{}'.format(
                                    nnsa.decode()),
                                         style={
                                             'width': '45%',
                                             'margin-right': '0.02em',
                                             'display': 'inline'
                                         })
                            ],
                                     style={
                                         'display': 'inline-block',
                                         'width': '10%'
                                     }),
                            html.P(funding_statement,
                                   style={
                                       'background-color': '#F5F5F5',
                                       'font-size': '1.5rem',
                                       'font-family': 'sans-serif',
                                       'display': 'inline-block',
                                       'width': '90%'
                                   })
                        ])
                ])
        ])

    @app.callback(Output('text-area', 'value'),
                  [Input('graph', 'mouseoverNodeData')])
    def update_node_data(node_data):
        if node_data is not None:
            label = str(node_data['label'])
            num = '\nneuron number: ' + str(node_data['id'])
            thresh = '\nthreshold: ' + str(node_data['threshold']) + '\n'
            value = label + num + thresh
            return value
        else:
            return ''

    @app.callback(Output('graph', 'elements'), [Input('slider', 'value')])
    def update_elements(X):
        elements_ls = []
        result_dict = results_dict(result, scaffold)
        if X in result_dict.keys():
            spiked_at_time = result_dict[X]
        else:
            spiked_at_time = []
        nodes, edges = _build_node_and_edge_lists(graph, pos, spiked_at_time)
        elements_ls = nodes + edges

        return elements_ls

    @app.callback(Output('summary', 'value'),
                  [Input('output_options', 'value')])
    def update_summary(output_node):
        result_dict = results_dict(result, scaffold)
        if output_node == '':
            return ''
        elif output_node == None:
            return ''
        for l in result_dict.values():
            if output_node in l:
                return 'Output node ' + str(output_node) + ' spiked.'
            else:
                value = 'Output node ' + str(output_node) + ' did not spike.'
        return value

    app.run_server(debug=debug)
    return


def _build_node_and_edge_lists(graph, pos, spiked_at_time):
    nodes = deque()
    edges = deque()

    for node in graph.nodes:
        node_data = {}
        node_position = {'x': pos[node][0], 'y': pos[node][1]}
        node_data['id'] = graph.nodes[node]['neuron_number']
        node_data['label'] = node
        node_data['threshold'] = graph.nodes[node]['threshold']
        node_dict = {
            'data': node_data,
            'position': node_position,
            'classes': 'spikes' if str(node) in spiked_at_time else 'nonspikes'
        }
        nodes.append(node_dict)

    for edge in graph.edges:

        source = edge[0]
        target = edge[1]

        edges.append({
            'data': {
                'source': graph.nodes[source]['neuron_number'],
                'target': graph.nodes[target]['neuron_number'],
                'label': str(source) + ' to ' + str(target),
                'weight': graph.edges[edge]['weight'],
                'delay': graph.edges[edge]['delay']
            }
        })
    return list(nodes), list(edges)


def _build_left_div(fig, elements_ls, stylesheet_list, time_step, max_t):
    return html.Div(
        className="six columns",
        children=[
            dcc.Graph(id='scaffold_graph', figure=fig),
            html.Div(className="twelve columns",
                     children=[
                         html.Label('scaffold.graph',
                                    style={
                                        'font-family': 'sans-serif',
                                        'font-size': '2.8rem',
                                        'background': '#FFFFFF',
                                        'margin-top': '15pt',
                                        'margin-left': '0rem',
                                        'display': 'block',
                                        'width': '100%'
                                    })
                     ]),
            cyto.Cytoscape(id='graph',
                           elements=elements_ls,
                           style={
                               'width': '100%',
                               'height': '400px',
                               'background': '#FFFFFF',
                               'margin-top': '45pt'
                           },
                           layout={'name': 'preset'},
                           autoRefreshLayout=False,
                           stylesheet=stylesheet_list),
            dcc.Textarea(
                id='text-area',
                draggable=True,
                placeholder='hover over a node to view its attributes',
                readOnly=True,
                value='',
                style={
                    'width': '100%',
                    'background': '#FFFFFF'
                }),
            html.P(children=[
                html.Div(className="twelve",
                         children=[
                             html.Label("timestep",
                                        style={
                                            'font-family': 'sans-serif',
                                            'font-size': '2.0rem',
                                            'background': '#FFFFFF',
                                            'margin-left': '0rem',
                                            'display': 'block',
                                            'margin-top': '0rem'
                                        })
                         ]),
                html.Div(className="twelve columns",
                         children=[
                             dcc.Slider(id='slider',
                                        marks=time_step,
                                        min=0,
                                        max=max_t,
                                        value=0,
                                        step=1,
                                        updatemode='drag')
                         ],
                         style={'margin-bottom': '5rem'})
            ])
        ])


def _build_right_div(output_options):
    return html.Div(
        className="six columns",
        children=[
            html.Div(className="twelve columns",
                     children=[
                         html.Label('decode results',
                                    style={
                                        'font-family': 'sans-serif',
                                        'font-size': '2.8rem',
                                        'background': '#FFFFFF',
                                        'margin-left': '0rem',
                                        'display': 'block'
                                    }),
                         html.Label('output neurons',
                                    style={
                                        'font-family': 'sans-serif',
                                        'font-size': '2.0rem',
                                        'background': '#FFFFFF',
                                        'margin-left': '0rem',
                                        'display': 'block'
                                    }),
                         dcc.Dropdown(id='output_options',
                                      options=output_options,
                                      value=''),
                         dcc.Textarea(id='summary',
                                      draggable=True,
                                      placeholder='view decoded results here',
                                      readOnly=True,
                                      value='',
                                      style={
                                          'width': '100%',
                                          'background': '#FFFFFF',
                                          'margin-bottom': '15pt'
                                      })
                     ]),
        ])


def _build_right_div_raster(output_options, raster_fig):
    return html.Div(
        className="six columns",
        children=[
            html.Div(className="twelve columns",
                     children=[
                         html.Label('decode results',
                                    style={
                                        'font-family': 'sans-serif',
                                        'font-size': '2.8rem',
                                        'background': '#FFFFFF',
                                        'margin-left': '0rem',
                                        'display': 'block'
                                    }),
                         html.Label('output neurons',
                                    style={
                                        'font-family': 'sans-serif',
                                        'font-size': '2.0rem',
                                        'background': '#FFFFFF',
                                        'margin-left': '0rem',
                                        'display': 'block'
                                    }),
                         dcc.Dropdown(id='output_options',
                                      options=output_options,
                                      value=''),
                         dcc.Textarea(id='summary',
                                      draggable=True,
                                      placeholder='view decoded results here',
                                      readOnly=True,
                                      value='',
                                      style={
                                          'width': '100%',
                                          'background': '#FFFFFF',
                                          'margin-bottom': '15pt'
                                      }),
                         dcc.Graph(id='raster_plot', figure=raster_fig)
                     ]),
        ])


def _build_right_div_expected(output_options, expected_fig):
    return html.Div(
        className="six columns",
        children=[
            html.Div(className="twelve columns",
                     children=[
                         html.Label('decode results',
                                    style={
                                        'font-family': 'sans-serif',
                                        'font-size': '2.8rem',
                                        'background': '#FFFFFF',
                                        'margin-left': '0rem',
                                        'display': 'block'
                                    }),
                         html.Label('output neurons',
                                    style={
                                        'font-family': 'sans-serif',
                                        'font-size': '2.0rem',
                                        'background': '#FFFFFF',
                                        'margin-left': '0rem',
                                        'display': 'block'
                                    }),
                         dcc.Dropdown(id='output_options',
                                      options=output_options,
                                      value=''),
                         dcc.Textarea(id='summary',
                                      draggable=True,
                                      placeholder='view decoded results here',
                                      readOnly=True,
                                      value='',
                                      style={
                                          'width': '100%',
                                          'background': '#FFFFFF',
                                          'margin-bottom': '15pt'
                                      }),
                         dcc.Graph(id='expected',
                                   figure=expected_fig,
                                   style={'margin-bottom': '2rem'})
                     ]),
        ])


def _build_right_div_raster_expected(output_options, raster_fig, expected_fig):
    return html.Div(
        className="six columns",
        children=[
            html.Div(className="twelve columns",
                     children=[
                         html.Label('decode results',
                                    style={
                                        'font-family': 'sans-serif',
                                        'font-size': '2.8rem',
                                        'background': '#FFFFFF',
                                        'margin-left': '0rem',
                                        'display': 'block'
                                    }),
                         html.Label('output neurons',
                                    style={
                                        'font-family': 'sans-serif',
                                        'font-size': '2.0rem',
                                        'background': '#FFFFFF',
                                        'margin-left': '0rem',
                                        'display': 'block'
                                    }),
                         dcc.Dropdown(id='output_options',
                                      options=output_options,
                                      value=''),
                         dcc.Textarea(id='summary',
                                      draggable=True,
                                      placeholder='view decoded results here',
                                      readOnly=True,
                                      value='',
                                      style={
                                          'width': '100%',
                                          'background': '#FFFFFF',
                                          'margin-bottom': '15pt'
                                      }),
                         dcc.Graph(id='expected',
                                   figure=expected_fig,
                                   style={'margin-bottom': '2rem'}),
                         dcc.Graph(id='raster_plot', figure=raster_fig)
                     ]),
        ])


def raster_plot(result, scaffold):
    """
    Raster plot of scaffold.graph
    
    Arguments:
        + result - Pandas DataFrame
        + scaffold - Fugu Scaffold
    """
    result = fill_results_from_graph(result, scaffold)

    print(result)
    times = []
    nodes = []
    i = 0
    for t in result['time']:
        times.append(int(t))
        nodes.append(int(result['neuron_number'][i]))
        i = i + 1

    plt.figure(figsize=(20, 10))
    plt.axis([0, times[-1] + 1, 0, scaffold.graph.number_of_nodes()])
    plt.plot(times, nodes, '.')
    plt.xlabel("Simulation time")
    plt.ylabel("Neuron index")
    plt.title("Raster Plot")

    plt.show()
    return


if __name__ == "__main__":
    import networkx as nx
    from fugu.scaffold import Scaffold
    from fugu.bricks.bricks import Vector_Input
    from fugu.bricks.utility_bricks import Copy, Dot
    from fugu.bricks.graph_bricks import Shortest_Path
    from fugu.bricks.stochastic_bricks import Threshold
    #, Copy, Dot, Threshold, Shortest_Path
    scaffold = Scaffold()
    print("Building Graph")
    #Small graph:
    scaffold.add_brick(Vector_Input(np.array([1, 0, 1]), coding='Raster'),
                       input_nodes='input')
    scaffold.add_brick(Copy())
    scaffold.add_brick(Dot([1, 0, 1], name='FirstDotOperator'), (1, 0))
    scaffold.add_brick(Dot([0, 0, 1], name='SecondDotOperator'), (1, 1))
    scaffold.add_brick(Threshold(1.25, name='LargerThanOneA'), (2, 0),
                       output=True)
    scaffold.add_brick(Threshold(1.25, name='LargerThanOneB'), (3, 0),
                       output=True)
    #Large graph:
    #    scaffold.add_brick(Vector_Input(np.array([1]), coding='Raster', name='Input0'), 'input' )
    #    target_graph = nx.generators.path_graph(5000)
    #    for edge in target_graph.edges:
    #        target_graph.edges[edge]['weight'] = 1.0
    #    scaffold.add_brick(Shortest_Path(target_graph,0))
    #
    print("Laying Bricks")
    scaffold.lay_bricks()
    #scaffold.summary()
    print()
    print('----------------------')
    print('Results:', end='\n\n')
    print("Evaluating")
    result = scaffold.evaluate(backend='ds_legacy',
                               max_runtime=100,
                               record_all=True)
    print("Done evaluating.")
    print("Building Layout")
    #setting pos is optional: also can set position with networkx layout generators
    pos = set_position(scaffold)

    print("Rendering graph")
    check_spikes = {
        0: {
            0: 1,
            1: 0
        },
        2: {
            0: 1
        },
        13: {
            0: 0,
            1: 0,
            2: 1
        },
        14: {
            0: 0,
            1: 0,
            2: 1
        }
    }
    graph_vis(result,
              scaffold,
              pos=pos,
              debug=False,
              check_spikes=check_spikes)
