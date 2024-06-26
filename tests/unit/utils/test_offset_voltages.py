import networkx as nx
import numpy as np
import pytest

from fugu.utils.optimization import offset_voltages


def create_test_graph(threshold=None, potential=None, reset_voltage=None):
    graph = nx.Graph()
    graph.add_node("a")

    if threshold is not None:
        graph.nodes["a"]["threshold"] = threshold
    if potential is not None:
        graph.nodes["a"]["potential"] = potential
    if reset_voltage is not None:
        graph.nodes["a"]["reset_voltage"] = reset_voltage

    return graph


@pytest.mark.parametrize(
    "threshold, reset_voltage, expected",
    [
        (1, 0, 1),
        (1, -1, 2),
        (1, 1, 0),
        (10, 1, 9),
        (10, 9, 1),
        (10, -1, 11),
        (-1, 0, -1),
        (-1, -1, 0),
        (-1, 1, -2),
        (-10, 1, -11),
        (-10, 9, -19),
        (-10, -1, -9),
        (0.1, 0, 0.1),
        (0.1, -1, 1.1),
        (0.1, 1, -0.9),
        (-0.1, 0, -0.1),
        (-0.1, -1, 0.9),
        (-0.1, 1, -1.1),
    ],
)
def test_zero_desired_reset_voltage(threshold, reset_voltage, expected):
    example_graph = create_test_graph(threshold, None, reset_voltage)
    example_graph = offset_voltages(example_graph, desired_reset_value=0)
    assert example_graph.nodes["a"]["threshold"] == expected


@pytest.mark.parametrize(
    "threshold, reset_voltage, desired_reset_voltage, expected",
    [
        (1, 0, 1, 2),
        (1, -1, 1, 3),
        (1, 1, 1, 1),
        (10, 1, 1, 10),
        (10, 9, 1, 2),
        (10, -1, 1, 12),
        (-1, 0, 1, 0),
        (-1, -1, 1, 1),
        (-1, 1, 1, -1),
        (-10, 1, 1, -10),
        (-10, 9, 1, -18),
        (-10, -1, 1, -8),
        (0.1, 0, 1, 1.1),
        (0.1, -1, 1, 2.1),
        (0.1, 1, 1, 0.1),
        (-0.1, 0, 1, 0.9),
        (-0.1, -1, 1, 1.9),
        (-0.1, 1, 1, -0.1),
    ],
)
def test_positive_desired_reset_voltage(threshold, reset_voltage, desired_reset_voltage, expected):
    example_graph = create_test_graph(threshold, None, reset_voltage)
    example_graph = offset_voltages(example_graph, desired_reset_value=desired_reset_voltage)
    assert example_graph.nodes["a"]["threshold"] == expected


@pytest.mark.parametrize(
    "threshold, reset_voltage, desired_reset_voltage, expected",
    [
        (1, 0, -1, 0),
        (1, -1, -1, 1),
        (1, 1, -1, -1),
        (10, 1, -1, 8),
        (10, 9, -1, 0),
        (10, -1, -1, 10),
        (-1, 0, -1, -2),
        (-1, -1, -1, -1),
        (-1, 1, -1, -3),
        (-10, 1, -1, -12),
        (-10, 9, -1, -20),
        (-10, -1, -1, -10),
        (0.1, 0, -1, -0.9),
        (0.1, -1, -1, 0.1),
        (0.1, 1, -1, -1.9),
        (-0.1, 0, -1, -1.1),
        (-0.1, -1, -1, -0.1),
        (-0.1, 1, -1, -2.1),
    ],
)
def test_negative_desired_reset_voltage(threshold, reset_voltage, desired_reset_voltage, expected):
    example_graph = create_test_graph(threshold, None, reset_voltage)
    example_graph = offset_voltages(example_graph, desired_reset_value=desired_reset_voltage)
    assert example_graph.nodes["a"]["threshold"] == expected


@pytest.mark.parametrize("reset_voltage, expected", [(0, 1), (1, 0), (-1, 2), (10, -9)])
def test_no_threshold(reset_voltage, expected):
    example_graph = create_test_graph(None, None, reset_voltage)
    assert "threshold" not in example_graph.nodes["a"]
    example_graph = offset_voltages(example_graph, desired_reset_value=0)
    assert "threshold" in example_graph.nodes["a"]
    assert example_graph.nodes["a"]["threshold"] == expected


@pytest.mark.parametrize("reset_voltage, expected", [(0, 0), (1, -1), (-1, 1), (10, -10)])
def test_no_potential(reset_voltage, expected):
    example_graph = create_test_graph(None, None, reset_voltage)
    assert "potential" not in example_graph.nodes["a"]
    example_graph = offset_voltages(example_graph, desired_reset_value=0)
    assert "potential" in example_graph.nodes["a"]
    assert example_graph.nodes["a"]["potential"] == expected
