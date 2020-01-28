import unittest
import math

import numpy as np
import networkx as nx

import fugu.bricks as BRICKS
from fugu.scaffold import Scaffold
from fugu.backends import ds_Backend, snn_Backend, pynn_Backend

from ..utilities import create_graph, create_weighted_graph

from ..base import BrickTest


DEBUG_REGISTERS = False 

class GraphTraversalBrickTests(BrickTest):
    def build_scaffold(self, input_values):
        graph, source_vertex, return_pred, use_edge_references = input_values
        scaffold = Scaffold()
        input_brick = BRICKS.Vector_Input(self.convert_input((graph, source_vertex)), coding='Raster', name='Input')
        traversal_brick = BRICKS.Graph_Traversal(
                                  graph,
                                  name="GraphTraversal",
                                  store_edge_references=use_edge_references,
                                  store_parent_info=return_pred)

        scaffold.add_brick(input_brick, 'input')
        scaffold.add_brick(traversal_brick, input_nodes=[(0, 0)], output=True)

        scaffold.lay_bricks()
        return scaffold

    def calculate_max_timesteps(self, input_values):
        graph = input_values[0]
        num_vertices = len(graph.nodes)
        weights = nx.get_edge_attributes(graph, 'weight').values()
        max_weight = max(weights) if len(weights) > 0 else 1
        return 2 * num_vertices * max_weight

    def check_spike_output(self, spikes, expected, scaffold):
        if type(spikes) is list:
            spikes = spikes[1]

        for node in scaffold.circuit.nodes:
            if scaffold.circuit.nodes[node]['name'] == 'GraphTraversal':
                traversal_brick = scaffold.circuit.nodes[node]['brick']

        graph = traversal_brick.target_graph
        use_edge_references = traversal_brick.store_edge_references
        return_path = traversal_brick.store_parent_info or use_edge_references

        predecessors = {v: -1 for v in graph.nodes}
        distance_table = {v: -1 for v in graph.nodes}
        start_time = 0.0

        names = list(scaffold.graph.nodes.data('name'))

        parent_registers = {}
        id_registers = {}
        for row in spikes.sort_values("time").itertuples():
            spikes += 1
            neuron_name = names[int(row.neuron_number)][0]

            neuron_props = scaffold.graph.nodes[neuron_name]
            if 'begin' in neuron_name:
                start_time = row.time
            elif 'is_vertex' in neuron_props:
                v = neuron_props['index'][0]
                distance_table[v] = row.time
            if return_path:
                if use_edge_references and 'is_edge_reference' in neuron_props:
                    u = neuron_props['from_vertex']
                    v = neuron_props['to_vertex']

                    predecessors[v] = u if u < predecessors[v] or predecessors[v] < 0 else predecessors[v]
                elif 'is_register' in neuron_props:
                    tag = neuron_props['register_tag']
                    if 'Parent' in neuron_props['register_name']:
                        if tag not in parent_registers:
                            parent_registers[tag] = ['0' for i in range(traversal_brick.register_size)]
                        parent_registers[tag][int(neuron_props['register_index'])] = '1'
                    if 'ID' in neuron_props['register_name']:
                        if tag not in id_registers:
                            id_registers[tag] = ['0' for i in range(traversal_brick.register_size)]
                        id_registers[tag][int(neuron_props['register_index'])] = '1'

        for v in distance_table:
            if distance_table[v] > -1:
                distance_table[v] -= start_time
                distance_table[v] /= 2.0

        for u, v in graph.edges():
            u_dist = distance_table[u]
            v_dist = distance_table[v]
            edge_data = graph.get_edge_data(u, v)
            edge_weight = edge_data['weight'] if 'weight' in edge_data else 1
            self.assertTrue(abs(u_dist - v_dist) <= edge_weight)

        if return_path:
            if not use_edge_references:
                # process register values
                id_to_node = {}
                for tag in id_registers:
                    id_to_node[''.join(id_registers[tag])] = scaffold.graph.nodes[tag]['index'][0]

                for tag in parent_registers:
                    vertex = scaffold.graph.nodes[tag]['index'][0]
                    pred = id_to_node[''.join(parent_registers[tag])]
                    predecessors[vertex] = pred

            for u in predecessors:
                v = predecessors[u]
                if v != -1:
                    u_dist = distance_table[u]
                    v_dist = distance_table[v]

                    self.assertTrue(u_dist != -1)
                    self.assertTrue(v_dist != -1)

                    edge_data = graph.get_edge_data(u, v)
                    if edge_data:
                        edge_weight = edge_data['weight'] if 'weight' in edge_data else 1
                    else:
                        edge_weight = 1

                    #print(u_dist, v_dist, edge_weight)
                    self.assertTrue(abs(u_dist - v_dist) <= edge_weight)

    def convert_input(self, input_values):
        graph, search_key = input_values
        spikes = [0] * len(graph.nodes)
        spikes[search_key] = 1
        return spikes

    @unittest.skipIf(DEBUG_REGISTERS, "Debugging registers")
    def test_bfs_random_gnp_level(self):
        graph = create_graph(20, 0.3, 3)
        self.basic_test((graph, 1, False, False), "")

    @unittest.skipIf(DEBUG_REGISTERS, "Debugging registers")
    def test_bfs_random_gnp_edge_references(self):
        graph = create_graph(20, 0.3, 3)
        self.basic_test((graph, 1, True, True), "")

    @unittest.skipIf(not DEBUG_REGISTERS, "Debugging registers")
    def test_bfs_random_gnp_edge_registers(self):
        graph = create_graph(20, 0.3, 3)
        self.basic_test((graph, 1, True, False), "")

    @unittest.skipIf(DEBUG_REGISTERS, "Debugging registers")
    def test_bfs_multi_run_levels(self):
        graph = create_graph(20, 0.3, 3)
        properties = []
        source_keys = [1, 5, 7]
        for key in source_keys:
            properties.append({'Input': {'spike_vector': self.convert_input((graph, key))}})
        self.run_property_test((graph, 1, False, False), properties, ["" for key in source_keys])

    @unittest.skipIf(DEBUG_REGISTERS, "Debugging registers")
    def test_bfs_multi_run_edge_references(self):
        graph = create_graph(20, 0.3, 3)
        properties = []
        source_keys = [1, 5, 7]
        for key in source_keys:
            properties.append({'Input': {'spike_vector': self.convert_input((graph, key))}})
        self.run_property_test((graph, 1, True, True), properties, ["" for key in source_keys])

    @unittest.skipIf(not DEBUG_REGISTERS, "Debugging registers")
    def test_bfs_multi_run_edge_registers(self):
        graph = create_graph(20, 0.3, 3)
        properties = []
        source_keys = [1, 5, 7]
        for key in source_keys:
            properties.append({'Input': {'spike_vector': self.convert_input((graph, key))}})
        self.run_property_test((graph, 1, True, False), properties, ["" for key in source_keys])

    @unittest.skipIf(DEBUG_REGISTERS, "Debugging registers")
    def test_sssp_random_gnp_distances(self):
        graph = create_weighted_graph(20, 0.3, 3)
        self.basic_test((graph, 1, False, False), "")

    @unittest.skipIf(DEBUG_REGISTERS, "Debugging registers")
    def test_sssp_random_gnp_edge_references(self):
        graph = create_weighted_graph(20, 0.3, 3)
        self.basic_test((graph, 1, True, True), "")

    @unittest.skipIf(not DEBUG_REGISTERS, "Debugging registers")
    def test_sssp_random_gnp_edge_registers(self):
        graph = create_weighted_graph(20, 0.3, 3)
        self.basic_test((graph, 1, True, False), "")

    @unittest.skipIf(DEBUG_REGISTERS, "Debugging registers")
    def test_sssp_multi_run_distances(self):
        graph = create_weighted_graph(20, 0.3, 3)
        properties = []
        source_keys = [1, 5, 7]
        for key in source_keys:
            properties.append({'Input': {'spike_vector': self.convert_input((graph, key))}})
        self.run_property_test((graph, 1, False, False), properties, ["" for key in source_keys])

    @unittest.skipIf(DEBUG_REGISTERS, "Debugging registers")
    def test_sssp_multi_run_edge_references(self):
        graph = create_weighted_graph(20, 0.3, 3)
        properties = []
        source_keys = [1, 5, 7]
        for key in source_keys:
            properties.append({'Input': {'spike_vector': self.convert_input((graph, key))}})
        self.run_property_test((graph, 1, True, True), properties, ["" for key in source_keys])

    @unittest.skipIf(not DEBUG_REGISTERS, "Debugging registers")
    def test_sssp_multi_run_edge_registers(self):
        graph = create_weighted_graph(20, 0.3, 3)
        properties = []
        source_keys = [1, 5, 7]
        for key in source_keys:
            properties.append({'Input': {'spike_vector': self.convert_input((graph, key))}})
        self.run_property_test((graph, 1, True, False), properties, ["" for key in source_keys])

    @unittest.skipIf(DEBUG_REGISTERS, "Debugging registers")
    def test_sssp_race_condition(self):
        graph = nx.DiGraph()
        graph.add_edge(0, 1, weight=1)
        graph.add_edge(1, 2, weight=1)
        graph.add_edge(2, 3, weight=1)
        graph.add_edge(3, 4, weight=1)
        graph.add_edge(4, 5, weight=1)
        graph.add_edge(5, 6, weight=1)
        graph.add_edge(6, 7, weight=1)
        graph.add_edge(7, 8, weight=1)
        graph.add_edge(8, 9, weight=1)
        graph.add_edge(0, 10, weight=10)
        graph.add_edge(1, 11, weight=9)
        graph.add_edge(2, 12, weight=8)
        graph.add_edge(3, 13, weight=7)
        graph.add_edge(4, 14, weight=6)
        graph.add_edge(5, 15, weight=5)
        graph.add_edge(6, 16, weight=4)
        graph.add_edge(7, 17, weight=3)
        graph.add_edge(8, 18, weight=2)
        graph.add_edge(9, 19, weight=1)
        self.basic_test((graph, 0, False, False), "")


class SnnGraphTraversalTests(GraphTraversalBrickTests, unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.backend = snn_Backend()


class DsGraphTraversalTests(GraphTraversalBrickTests, unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.backend = ds_Backend()


class PynnSpinnakerGraphTraversalTests(GraphTraversalBrickTests, unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.backend = pynn_Backend()
        self.backend_args['backend'] = 'spinnaker'


class PynnBrianGraphTraversalTests(GraphTraversalBrickTests, unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.backend = pynn_Backend()
        self.backend_args['backend'] = 'brian'
