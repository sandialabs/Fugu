import pytest
import networkx as nx
import numpy as np

from fugu.utils.stats import get_max_magnitude_synapse_values, get_max_magnitude_neuron_values

def build_graph_from_edges(edge_list=[], weight_list=[]):
    graph = nx.DiGraph()

    for edge, weight in zip(edge_list, weight_list):
        graph.add_edge(edge[0], edge[1])
        if weight is not None:
            graph.edges[edge]['weight'] = weight

    return graph

def build_graph_from_nodes(node_list=[], property_list=[]):
    graph = nx.DiGraph()

    for node, properties in zip(node_list, property_list):
        graph.add_node(node)
        for prop in properties:
            if properties[prop] is not None:
                graph.nodes[node][prop] = properties[prop]

    return graph


@pytest.mark.parametrize("weights, expected", [
                            ((2, 3, 4, 5, 6, 7, 8, 9), 9),
                            ((8, 9), 9),
                            ((1, 0), 1),
                            ((1, 10), 10),
                            ((-1, 10), 10),
                            ((-1, 0), 1),
                            ((-1, -10, -2), 10),
                            ((1, -10, 2), 10),
                            ]
                            )
def test_max_weight(weights, expected):
    edge_list = []
    for index in range(len(weights)):
        edge_list.append((index, index+1))
    graph = build_graph_from_edges(edge_list, weights)

    calculated_values = get_max_magnitude_synapse_values(graph, ['weight'])
    assert expected == calculated_values['weight']

@pytest.mark.parametrize("weights, default_value, expected", [
                            ((-2, 3, None, 5, 6, 7, 8, 9), 1, 9),
                            ((-2, 3, None, 5, 6, 7, 8, 9), 10, 10),
                            ((-2, 3, None, 5, 6, 7, 8, 9), -3, 9),
                            ((-2, 3, None, 5, 6, 7, 8, -9), -3, 9),
                            ((-2, 3, None, 5, 6, 7, 8, 9), -13, 13),
                            ]
                            )
def test_missing_weight(weights, default_value, expected):
    edge_list = []
    for index in range(len(weights)):
        edge_list.append((index, index+1))
    graph = build_graph_from_edges(edge_list, weights)

    calculated_values = get_max_magnitude_synapse_values(graph, ['weight'], default_values={'weight': default_value})
    assert expected == calculated_values['weight']

@pytest.mark.parametrize("thresholds, expected", [
                            ((2, 3, 4, 5, 6, 7, 8, 9), 9),
                            ((8, 9), 9),
                            ((1, 0), 1),
                            ((1, 10), 10),
                            ((-1, 10), 10),
                            ((-1, 0), 1),
                            ((-1, -10, -2), 10),
                            ((1, -10, 2), 10),
                            ]
                            )
def test_max_threshold(thresholds, expected):
    node_list = []
    property_list = []
    for index in range(len(thresholds)):
        node_list.append(index)
        property_list.append({'threshold': thresholds[index]})
    graph = build_graph_from_nodes(node_list, property_list)

    calculated_values = get_max_magnitude_neuron_values(graph, ['threshold'])
    assert expected == calculated_values['threshold']

@pytest.mark.parametrize("thresholds, default_value, expected", [
                            ((-2, 3, None, 5, 6, 7, 8, 9), 1, 9),
                            ((-2, 3, None, 5, 6, 7, 8, 9), 10, 10),
                            ((-2, 3, None, 5, 6, 7, 8, 9), -3, 9),
                            ((-2, 3, None, 5, 6, 7, 8, -9), -3, 9),
                            ((-2, 3, None, 5, 6, 7, 8, 9), -13, 13),
                            ]
                            )
def test_missing_threshold(thresholds, default_value, expected):
    node_list = []
    property_list = []
    for index in range(len(thresholds)):
        node_list.append(index)
        property_list.append({'threshold': thresholds[index]})
    graph = build_graph_from_nodes(node_list, property_list)

    calculated_values = get_max_magnitude_neuron_values(graph, ['threshold'], default_values={'threshold': default_value})
    assert expected == calculated_values['threshold']