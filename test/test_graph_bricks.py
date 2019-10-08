import unittest
import math
import random
import numpy as np

import networkx as nx
from networkx.generators.random_graphs import fast_gnp_random_graph

import fugu
import fugu.bricks as BRICKS
from fugu import Scaffold

random.seed(3)
def create_graph(size, p, seed):
    G = fast_gnp_random_graph(size, p, seed=seed)
    return G

def create_weighted_graph(size, p, seed):
    G = fast_gnp_random_graph(size, p, seed=seed)
    for (u,v) in G.edges():
        G.edges[u,v]['weight'] = random.randint(1,10)
    return G

class GraphBrickTests:
    backend = None
    backend_args = {}

    def get_spike_input(self, search_key):
        spikes = [0] * search_key
        spikes.append(1)
        return spikes

    def evaluate_bfs_graph(self, graph, search_key, debug=False):
        scaffold = Scaffold()

        bfs_input = BRICKS.Vector_Input(self.get_spike_input(search_key), coding='Raster', name='BFSInput')
        bfs_brick = BRICKS.Breadth_First_Search(graph, name="BFS")

        scaffold.add_brick(bfs_input, 'input')
        scaffold.add_brick(bfs_brick, output=True)

        scaffold.lay_bricks()
        if debug:
            scaffold.summary(verbose=2)

        results = scaffold.evaluate(backend=self.backend, max_runtime=len(graph.nodes) * 2, backend_args=self.backend_args)
        bfs_levels = {}
        bfs_names = list(scaffold.graph.nodes.data('name'))

        curr_level = 0
        curr_time = 0.0
        for row in results.sort_values("time").itertuples():
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

        for edge in graph.edges():
            u = edge[0]
            v = edge[1]
            u_level = bfs_levels[u]
            v_level = bfs_levels[v]
            self.assertTrue(abs(u_level - v_level) <= 1)

    def evaluate_sssp_graph(self, graph, search_key, return_path):
        scaffold = Scaffold()

        sssp_input = BRICKS.Vector_Input(self.get_spike_input(search_key), coding='Raster', name='SSSPInput')
        sssp_brick = BRICKS.Shortest_Path(graph, name="SSSP", return_path=return_path)

        scaffold.add_brick(sssp_input, 'input')
        scaffold.add_brick(sssp_brick, output=True)

        scaffold.lay_bricks()
        results = scaffold.evaluate(backend=self.backend, max_runtime=len(graph.nodes) * 30, backend_args=self.backend_args)
        sssp_pred = {v:-1 for v in graph.nodes}
        sssp_table = {v:-1 for v in graph.nodes}
        sssp_start_time = 0.0

        sssp_names = list(scaffold.graph.nodes.data('name'))
        sssp_spikes = 0
        for row in results.itertuples():
            sssp_spikes += 1
            neuron_name = sssp_names[int(row.neuron_number)][0]

            neuron_props = scaffold.graph.nodes[neuron_name]
            if 'begin' in neuron_name:
                sssp_start_time = row.time
            elif 'is_vertex' in neuron_props:
                v = neuron_props['index'][0]
                sssp_table[v] = row.time
            if return_path:
                if 'is_edge_reference' in neuron_props:
                    u = neuron_props['from_vertex']
                    v = neuron_props['to_vertex']

                    sssp_pred[v] = u if u < sssp_pred[v] or sssp_pred[v] < 0 else sssp_pred[v]

        for v in sssp_table:
            if sssp_table[v] > -1:
                sssp_table[v] -= sssp_start_time
                sssp_table[v] /= 2.0

        for u,v in graph.edges():
            u_dist = sssp_table[u]
            v_dist = sssp_table[v]
            edge_weight = graph.get_edge_data(u,v)['weight']
            self.assertTrue(abs(u_dist - v_dist) <= edge_weight)

        if return_path:
            for u in sssp_pred:
                v = sssp_pred[u]
                if v > -1:
                    u_dist = sssp_table[u]
                    v_dist = sssp_table[v]

                    self.assertTrue(u_dist > -1)
                    self.assertTrue(v_dist > -1)

                    edge_weight = graph.get_edge_data(u,v)['weight']

                    self.assertTrue(abs(u_dist - v_dist) <= edge_weight)

    def test_bfs_random_gnp(self):
        self.evaluate_bfs_graph(create_graph(20, 0.3, 3), 1)

    def test_sssp_race_condition(self): 
        graph = nx.DiGraph()
        graph.add_edge(0,1,weight=1)
        graph.add_edge(1,2,weight=1)
        graph.add_edge(2,3,weight=1)
        graph.add_edge(3,4,weight=1)
        graph.add_edge(4,5,weight=1)
        graph.add_edge(5,6,weight=1)
        graph.add_edge(6,7,weight=1)
        graph.add_edge(7,8,weight=1)
        graph.add_edge(8,9,weight=1)
        graph.add_edge(0,10,weight=10)
        graph.add_edge(1,11,weight=9)
        graph.add_edge(2,12,weight=8)
        graph.add_edge(3,13,weight=7)
        graph.add_edge(4,14,weight=6)
        graph.add_edge(5,15,weight=5)
        graph.add_edge(6,16,weight=4)
        graph.add_edge(7,17,weight=3)
        graph.add_edge(8,18,weight=2)
        graph.add_edge(9,19,weight=1)
        self.evaluate_sssp_graph(graph, 0, False)

    def test_sssp_random_gnp_dist(self):
        graph = create_weighted_graph(20, 0.3, 3)
        self.evaluate_sssp_graph(graph, 1, False)

    def test_sssp_random_gnp_full(self):
        graph = create_weighted_graph(20, 0.3, 3)
        self.evaluate_sssp_graph(graph, 1, True)

class SnnBackendGraphTests(unittest.TestCase, GraphBrickTests):
    @classmethod
    def setUpClass(self):
        self.backend = 'snn'

class DsBackendGraphTests(unittest.TestCase, GraphBrickTests):
    @classmethod
    def setUpClass(self):
        self.backend = 'ds'

class PynnBrianBackendGraphTests(unittest.TestCase, GraphBrickTests):
    @classmethod
    def setUpClass(self):
        self.backend = 'pynn'
        self.backend_args['backend'] = 'brian'

class PynnSpinnakerBackendGraphTests(unittest.TestCase, GraphBrickTests):
    @classmethod
    def setUpClass(self):
        self.backend = 'pynn'
        self.backend_args['backend'] = 'spinnaker'
