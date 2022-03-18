import unittest
import math

import numpy as np
import networkx as nx

import fugu.bricks as BRICKS
from fugu.scaffold import Scaffold
from fugu.backends import ds_Backend, snn_Backend, pynn_Backend

from ..utilities import create_graph, create_weighted_graph

from ..base import BrickTest


class RegisterGraphTraversalBrickTests(BrickTest):
    def build_scaffold(self, input_values):
        graph, source_vertex, return_pred = input_values
        scaffold = Scaffold()
        input_brick = BRICKS.Vector_Input(self.convert_input(
            (graph, source_vertex)),
                                          coding='Raster',
                                          name='Input')
        traversal_brick = BRICKS.RegisterGraphTraversal(
            graph, name="RGT", store_parent_info=return_pred)

        scaffold.add_brick(input_brick, 'input')
        scaffold.add_brick(traversal_brick, input_nodes=[(0, 0)], output=True)

        scaffold.lay_bricks()
        return scaffold

    def calculate_max_timesteps(self, input_values):
        graph = input_values[0]
        num_vertices = len(graph.nodes)
        weights = nx.get_edge_attributes(graph, 'weight').values()
        max_weight = max(weights) if len(weights) > 0 else 1
        time_factor = 5 * max_weight if max_weight > 1 else 7
        return time_factor * num_vertices

    def check_spike_output(self, spikes, expected, scaffold):
        if type(spikes) is list:
            spikes = spikes[1]

        time_scale_factor = 1
        for node in scaffold.circuit.nodes:
            if scaffold.circuit.nodes[node]['name'] == 'RGT':
                traversal_brick = scaffold.circuit.nodes[node]['brick']
                time_scale_factor = traversal_brick.metadata[
                    'timescale_factor']

        graph = traversal_brick.target_graph
        return_path = traversal_brick.store_parent_info

        node_indices = traversal_brick.node_indices
        index_to_node = {}
        for node in node_indices:
            index_to_node[node_indices[node]] = node
            if self.debug:
                print("Node: {}, index: {}".format(node, node_indices[node]))

        predecessors_indices = {v: -1 for v in graph.nodes}
        predecessors = {v: -1 for v in graph.nodes}
        distance_table = {v: -1 for v in graph.nodes}
        start_time = 1.0

        names = list(scaffold.graph.nodes.data('name'))
        hit_count = 0

        if self.debug:
            print("Neuron, time")
        for row in spikes.sort_values("time").itertuples():
            spikes += 1
            neuron_name = names[int(row.neuron_number)][0]
            if self.debug:
                print(neuron_name, row.time)

            neuron_props = scaffold.graph.nodes[neuron_name]
            if 'is_vertex' in neuron_props:
                v = neuron_props['node']
                distance_table[v] = row.time
                hit_count += 1
            elif return_path and 'bit_position' in neuron_props:
                node = neuron_props['label']
                position = neuron_props['bit_position']
                if predecessors_indices[node] < 0:
                    predecessors_indices[node] = 2**position
                else:
                    predecessors_indices[node] += 2**position

        if nx.is_strongly_connected(graph):
            if self.debug:
                print(hit_count, len(graph.nodes))
            self.assertTrue(hit_count == len(graph.nodes))

        if return_path:
            for node in predecessors_indices:
                if self.debug:
                    print('node: {}, pred_index: {}'.format(
                        node, predecessors_indices[node]))
                pred_index = predecessors_indices[node]
                if pred_index > -1:
                    predecessors[node] = index_to_node[pred_index]
            if self.debug:
                for node in predecessors:
                    print("u {}, u index {}, pred {}, pred index {}".format(
                        node, node_indices[node], predecessors[node],
                        predecessors_indices[node]))

        for v in distance_table:
            if distance_table[v] > -1:
                distance_table[v] -= start_time
                distance_table[v] /= time_scale_factor
                if self.debug:
                    print(v, distance_table[v])

        for u, v, data in graph.edges(data=True):
            u_dist = distance_table[u]
            v_dist = distance_table[v]
            edge_weight = data['weight'] if 'weight' in data else 1
            if self.debug:
                print(u, v, u_dist, v_dist, edge_weight, data)
            if u_dist > -1 and v_dist > -1:
                self.assertTrue((v_dist - u_dist) <= edge_weight)

        if return_path:
            for u in predecessors:
                v = predecessors[u]
                if self.debug:
                    print("u, pred: {}, {}".format(u, v))
                if v != -1:
                    u_dist = distance_table[u]
                    v_dist = distance_table[v]

                    edge_data = graph.get_edge_data(v, u)
                    if edge_data:
                        edge_weight = edge_data[
                            'weight'] if 'weight' in edge_data else 1
                    else:
                        edge_weight = 1

                    if self.debug:
                        print(u, v, u_dist, v_dist, edge_weight, edge_data)

                    self.assertTrue(u_dist != -1)
                    self.assertTrue(v_dist != -1)
                    self.assertTrue((v_dist - u_dist) <= edge_weight)

    def convert_input(self, input_values):
        graph, search_key = input_values
        spikes = [0] * len(graph.nodes)
        spikes[search_key] = 1
        return spikes

    #@unittest.skip('')
    def test_bfs_random_gnp_level(self):
        graph = create_graph(15, 0.3, 3)
        self.basic_test((graph, 1, False), "")

    #@unittest.skip('')
    def test_bfs_random_gnp_store_parents(self):
        graph = create_graph(15, 0.3, 3)
        self.basic_test((graph, 1, True), "")

    #@unittest.skip('')
    def test_bfs_multi_run_levels(self):
        graph = create_graph(15, 0.3, 3)
        properties = []
        source_keys = [1, 5, 7]
        for key in source_keys:
            properties.append(
                {'Input': {
                    'spike_vector': self.convert_input((graph, key))
                }})
        self.run_property_test((graph, 1, False), properties,
                               ["" for key in source_keys])

    #@unittest.skip('')
    def test_bfs_multi_run_store_parents(self):
        graph = create_graph(15, 0.3, 3)
        properties = []
        source_keys = [1, 5, 1]
        for key in source_keys:
            properties.append(
                {'Input': {
                    'spike_vector': self.convert_input((graph, key))
                }})
        self.run_property_test((graph, 1, True), properties,
                               ["" for key in source_keys])

    #@unittest.skip('')
    def test_sssp_random_gnp_distances(self):
        graph = create_weighted_graph(15, 0.3, 3)
        self.basic_test((graph, 1, False), "")

    #@unittest.skip('')
    def test_sssp_random_gnp_store_parents(self):
        graph = create_weighted_graph(15, 0.3, 3)
        self.basic_test((graph, 1, True), "")

    #@unittest.skip('')
    def test_sssp_multi_run_distances(self):
        graph = create_weighted_graph(15, 0.3, 3)
        properties = []
        source_keys = [1, 5, 7]
        for key in source_keys:
            properties.append(
                {'Input': {
                    'spike_vector': self.convert_input((graph, key))
                }})
        self.run_property_test((graph, 1, False), properties,
                               ["" for key in source_keys])

    #@unittest.skip('')
    def test_sssp_multi_run_store_parents(self):
        graph = create_weighted_graph(15, 0.3, 3)
        properties = []
        source_keys = [1, 5, 1]
        for key in source_keys:
            properties.append(
                {'Input': {
                    'spike_vector': self.convert_input((graph, key))
                }})
        self.run_property_test((graph, 1, True), properties,
                               ["" for key in source_keys])

    #@unittest.skip('')
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
        self.basic_test((graph, 0, True), "")


class SnnRegisterGraphTraversalTests(RegisterGraphTraversalBrickTests,
                                     unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.backend = snn_Backend()


class DsRegisterGraphTraversalTests(RegisterGraphTraversalBrickTests,
                                    unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.backend = ds_Backend()


class PynnSpinnakerRegisterGraphTraversalTests(
        RegisterGraphTraversalBrickTests, unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.backend = pynn_Backend()
        self.backend_args['backend'] = 'spinnaker'


class PynnBrianRegisterGraphTraversalTests(RegisterGraphTraversalBrickTests,
                                           unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.backend = pynn_Backend()
        self.backend_args['backend'] = 'brian'
        self.backend_args[
            'single_fire'] = True  # this is kinda hacky but necessary because pynn-brian is wonky
