import unittest
import math
import numpy as np

import networkx as nx

import fugu
import fugu.bricks as BRICKS
from fugu import Scaffold
from fugu import ds_Backend, snn_Backend, pynn_Backend

from utilities import create_graph, create_weighted_graph


class GraphBrickTests:
    def build_scaffold(self, (graph, return_pred)):
        scaffold = Scaffold()
        bfs_input = BRICKS.Vector_Input(len(graph.nodes()), coding='Raster', name='BFSInput')
        bfs_brick = BRICKS.Graph_Traversal(graph, name="BFS-Edge-Reference", store_edge_references=return_pred)

        scaffold.add_brick(bfs_input, 'input')
        scaffold.add_brick(bfs_brick, output=True)

        scaffold.lay_bricks()
        return scaffold

    def calculate_max_timesteps(self, search_key):
        pass

    def check_spike_output(self, spikes, expected, scaffold):
        pass

    def convert_input(self, search_key):
        spikes = [0] * search_key
        spikes.append(1)
        return spikes

    def get_spike_input(self, search_key):
        pass

    def evaluate_bfs_graph(self, graph, search_keys, return_pred=False, debug=False):
        # Edge reference version
        scaffold = Scaffold()

        is_multi_run = False
        if len(search_keys) > 1:
            is_multi_run = True
            spike_inputs = []
            for search_key in search_keys:
                spike_inputs.append([0 for i in range(len(graph.nodes))])
                spike_inputs[-1][search_key] = 1
        else:
            spike_inputs = self.get_spike_input(search_keys[0])

        bfs_input = BRICKS.Vector_Input(spike_inputs, coding='Raster', name='BFSInput', multi_run_inputs=is_multi_run)
        bfs_brick = BRICKS.Graph_Traversal(graph, name="BFS-Edge-Reference", store_edge_references=return_pred)

        scaffold.add_brick(bfs_input, 'input')
        scaffold.add_brick(bfs_brick, output=True)

        scaffold.lay_bricks()
        if debug:
            scaffold.summary(verbose=2)

        results = scaffold.evaluate(
                             backend=self.backend,
                             max_runtime=len(graph.nodes) * 2,
                             backend_args=self.backend_args,
                             )

        def process_run(spikes, pred_method='edge'):
            bfs_levels = {}
            bfs_pred = {}
            bfs_names = list(scaffold.graph.nodes.data('name'))

            curr_level = 0
            curr_time = 0.0

            use_edge_references = pred_method == 'edge'
            parent_registers = {}
            id_registers = {}
            for row in spikes.sort_values("time").itertuples():
                neuron_name = bfs_names[int(row.neuron_number)][0]

                neuron_props = scaffold.graph.nodes[neuron_name]
                if debug:
                    print(neuron_name, row.time)

                if 'is_vertex' in neuron_props:
                    if row.time > curr_time:
                        curr_level += 1
                        curr_time = row.time
                    vertex = neuron_props['index'][0]

                    self.assertFalse(vertex in bfs_levels)

                    bfs_levels[vertex] = curr_level

                if use_edge_references and 'is_edge_reference' in neuron_props:
                    u = neuron_props['from_vertex']
                    v = neuron_props['to_vertex']
                    bfs_pred[v] = u
                elif 'is_register' in neuron_props:
                    tag = neuron_props['register_tag']
                    if 'Parent' in neuron_props['register_name']:
                        if tag not in parent_registers:
                            parent_registers[tag] = ['0' for i in range(bfs_brick.register_size)]
                        parent_registers[tag][int(neuron_props['register_index'])] = '1'
                    elif 'ID' in neuron_props['register_name']:
                        if tag not in id_registers:
                            id_registers[tag] = ['0' for i in range(bfs_brick.register_size)]
                        id_registers[tag][int(neuron_props['register_index'])] = '1'

            if not use_edge_references:
                # process register values
                id_to_node = {}
                for tag in id_registers:
                    id_to_node[''.join(id_registers[tag])] = tag

                for tag in parent_registers:
                    bfs_pred[tag] = id_to_node[''.join(parent_registers[tag])]

            for edge in graph.edges():
                u = edge[0]
                v = edge[1]
                u_level = bfs_levels[u]
                v_level = bfs_levels[v]
                self.assertTrue(abs(u_level - v_level) <= 1)

            if return_pred:
                for u in bfs_pred:
                    v = bfs_pred[u]
                    if v > -1:
                        u_level = bfs_levels[u]
                        v_level = bfs_levels[v]
                        self.assertTrue(abs(u_level - v_level) == 1)
                        self.assertTrue((u, v) in graph.edges() or (v, u) in graph.edges())

        if is_multi_run:
            for result in results:
                process_run(result)
        else:
            process_run(results)

        # Register version
        # "reset"
        scaffold = Scaffold()

        bfs_brick = BRICKS.Graph_Traversal(graph, name="BFS-Register", store_parent_info=return_pred)

        scaffold.add_brick(bfs_input, 'input')
        scaffold.add_brick(bfs_brick, output=True)

        scaffold.lay_bricks()
        if debug:
            scaffold.summary(verbose=2)

        results = scaffold.evaluate(
                             backend=self.backend,
                             max_runtime=len(graph.nodes) * 2,
                             backend_args=self.backend_args,
                             )

        if is_multi_run:
            for result in results:
                process_run(result)
        else:
            process_run(results)

    def evaluate_sssp_graph(self, graph, search_keys, return_path):
        # Edge reference
        scaffold = Scaffold()

        is_multi_run = False
        if len(search_keys) > 1:
            is_multi_run = True
            spike_inputs = []
            for search_key in search_keys:
                spike_inputs.append([0 for i in range(len(graph.nodes))])
                spike_inputs[-1][search_key] = 1
        else:
            spike_inputs = self.get_spike_input(search_keys[0])

        sssp_input = BRICKS.Vector_Input(
                              spike_inputs,
                              coding='Raster',
                              name='SSSPInput',
                              multi_run_inputs=is_multi_run,
                              )
        sssp_brick = BRICKS.Graph_Traversal(graph, name="SSSP", store_edge_references=return_path)

        scaffold.add_brick(sssp_input, 'input')
        scaffold.add_brick(sssp_brick, output=True)

        scaffold.lay_bricks()
        results = scaffold.evaluate(
                             backend=self.backend,
                             max_runtime=len(graph.nodes) * 30,
                             backend_args=self.backend_args,
                             )

        def process_run(spikes, pred_method='edge'):
            sssp_pred = {v: -1 for v in graph.nodes}
            sssp_table = {v: -1 for v in graph.nodes}
            sssp_start_time = 0.0

            sssp_names = list(scaffold.graph.nodes.data('name'))
            sssp_spikes = 0

            use_edge_references = pred_method == 'edge'
            parent_registers = {}
            id_registers = {}
            for row in spikes.sort_values("time").itertuples():
                sssp_spikes += 1
                neuron_name = sssp_names[int(row.neuron_number)][0]

                neuron_props = scaffold.graph.nodes[neuron_name]
                if 'begin' in neuron_name:
                    sssp_start_time = row.time
                elif 'is_vertex' in neuron_props:
                    v = neuron_props['index'][0]
                    sssp_table[v] = row.time
                if return_path:
                    if use_edge_references and 'is_edge_reference' in neuron_props:
                        u = neuron_props['from_vertex']
                        v = neuron_props['to_vertex']

                        sssp_pred[v] = u if u < sssp_pred[v] or sssp_pred[v] < 0 else sssp_pred[v]
                    elif 'is_register' in neuron_props:
                        tag = neuron_props['register_tag']
                        if 'Parent' in neuron_props['register_name']:
                            if tag not in parent_registers:
                                parent_registers[tag] = ['0' for i in range(sssp_brick.register_size)]
                            parent_registers[tag][int(neuron_props['register_index'])] = '1'
                        elif 'ID' in neuron_props['register_name']:
                            if tag not in id_registers:
                                id_registers[tag] = ['0' for i in range(sssp_brick.register_size)]
                            id_registers[tag][int(neuron_props['register_index'])] = '1'

            for v in sssp_table:
                if sssp_table[v] > -1:
                    sssp_table[v] -= sssp_start_time
                    sssp_table[v] /= 2.0

            for u, v in graph.edges():
                u_dist = sssp_table[u]
                v_dist = sssp_table[v]
                edge_weight = graph.get_edge_data(u, v)['weight']
                self.assertTrue(abs(u_dist - v_dist) <= edge_weight)

            if return_path:
                if not use_edge_references:
                    # process register values
                    id_to_node = {}
                    for tag in id_registers:
                        id_to_node[''.join(id_registers[tag])] = tag

                    for tag in parent_registers:
                        sssp_pred[tag] = id_to_node[''.join(parent_registers[tag])]

                for u in sssp_pred:
                    v = sssp_pred[u]
                    if v > -1:
                        u_dist = sssp_table[u]
                        v_dist = sssp_table[v]

                        self.assertTrue(u_dist > -1)
                        self.assertTrue(v_dist > -1)

                        edge_weight = graph.get_edge_data(u, v)['weight']

                        self.assertTrue(abs(u_dist - v_dist) <= edge_weight)

        if is_multi_run:
            for result in results:
                process_run(result)
        else:
            process_run(results)

        # Register Version
        scaffold = Scaffold()

        sssp_brick = BRICKS.Graph_Traversal(graph, name="SSSP", store_parent_info=return_path)

        scaffold.add_brick(sssp_input, 'input')
        scaffold.add_brick(sssp_brick, output=True)

        scaffold.lay_bricks()
        results = scaffold.evaluate(
                             backend=self.backend,
                             max_runtime=len(graph.nodes) * 30,
                             backend_args=self.backend_args,
                             )

        if is_multi_run:
            for result in results:
                process_run(result)
        else:
            process_run(results)

    def test_bfs_random_gnp_levels(self):
        self.evaluate_bfs_graph(create_graph(20, 0.3, 3), [1], False)

    def test_bfs_random_gnp_full(self):
        self.evaluate_bfs_graph(create_graph(20, 0.3, 3), [1], True)

    def test_bfs_multi_run_levels(self):
        self.evaluate_bfs_graph(create_graph(20, 0.3, 3), [1, 5, 7], False)

    def test_bfs_multi_run_full(self):
        self.evaluate_bfs_graph(create_graph(20, 0.3, 3), [1, 5, 7], True)

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
        self.evaluate_sssp_graph(graph, [0], False)

    def test_sssp_random_gnp_dist(self):
        graph = create_weighted_graph(20, 0.3, 3)
        self.evaluate_sssp_graph(graph, [1], False)

    def test_sssp_random_gnp_full(self):
        graph = create_weighted_graph(20, 0.3, 3)
        self.evaluate_sssp_graph(graph, [1], True)

    def test_sssp_multi_run_dist(self):
        graph = create_weighted_graph(20, 0.3, 3)
        self.evaluate_sssp_graph(graph, [1, 5, 7], False)

    def test_sssp_multi_run_full(self):
        graph = create_weighted_graph(20, 0.3, 3)
        self.evaluate_sssp_graph(graph, [1, 5, 7], True)


class SnnGraphTests(GraphBrickTests, unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.backend = snn_Backend()


class DsGraphTests(GraphBrickTests, unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.backend = ds_Backend()


class PynnSpinnakerGraphTests(GraphBrickTests, unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.backend = pynn_Backend()
        self.backend_args['backend'] = 'spinnaker'
