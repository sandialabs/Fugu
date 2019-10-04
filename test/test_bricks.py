import abc
import sys
if sys.version_info >= (3, 4):
    ABC = abc.ABC
else:
    ABC = abc.ABCMeta('ABC', (), {'__slots__': ()})
from abc import abstractmethod

import unittest
import math
import random
import numpy as np

import networkx as nx
from networkx.generators.random_graphs import fast_gnp_random_graph

import fugu
import fugu.bricks as BRICKS
from fugu import Scaffold

BACKEND = 'pynn'
BACKEND_ARGS = {}
BACKEND_ARGS['backend'] = 'brian'

def AssertValuesAreClose(value1, value2, tolerance = 0.0001):
    if not math.isclose(value1, value2, abs_tol=tolerance):
        raise AssertionError('Values {} and {} are not close'.format(value1, value2))

random.seed(3)
def create_graph(size, p, seed):
    G = fast_gnp_random_graph(size, p, seed=seed)
    return G

def create_weighted_graph(size, p, seed):
    G = fast_gnp_random_graph(size, p, seed=seed)
    for (u,v) in G.edges():
        G.edges[u,v]['weight'] = random.randint(1,10)
    return G

class BrickTest(unittest.TestCase, ABC):
    def setUp(self):
        super(BrickTest, self).setUp()
        self.scaffold = Scaffold()

    @abstractmethod
    def process_results(self, results):
        pass

    def evaluate_scaffold(self, max_time):
        self.scaffold.lay_bricks()
        results = self.scaffold.evaluate(backend=BACKEND, max_runtime=max_time, backend_args=BACKEND_ARGS)
        return self.process_results(results)

class LISTest(BrickTest):
    def process_results(self, results):
        graph_names = list(self.scaffold.graph.nodes.data('name'))
        lis = 0
        for row in results.itertuples():
            neuron_name = graph_names[int(row.neuron_number)][0]
            if "Main" in neuron_name:
                level = int(neuron_name.split("_")[1])
                if level > lis:
                    lis = level
        return lis
    
    def get_spike_input(self, sequence):
        num_in_sequence = len(sequence)
        max_time = max(sequence)
        spike_times = [[0] * (max_time + 1) for i in range(num_in_sequence)]
        for i, time in enumerate(sequence):
            spike_times[i][time] = 1
        return spike_times

    def evaluate_sequence(self, sequence, expected): 
        vector_brick = BRICKS.Vector_Input(self.get_spike_input(sequence), coding='Raster', name='Input0', time_dimension=True)
        self.scaffold.add_brick(vector_brick, 'input')
        self.scaffold.add_brick(BRICKS.LIS(len(sequence), name="LIS"), output=True)
        answer = self.evaluate_scaffold(100)
        self.assertEqual(expected, answer)

    def test_strictly_increasing(self):
        sequence = [1,2,3,4,5,6,7,8,9,10]
        self.evaluate_sequence(sequence,len(sequence))

    def test_strictly_decreasing(self):
        sequence = [10,10,5,5,1]
        self.evaluate_sequence(sequence,1)

    def test_general1(self):
        sequence = [5, 10, 2, 3, 4]
        self.evaluate_sequence(sequence,3)

    def test_general2(self):
        sequence = [1,4,8,6,2,7,19,13,14]
        self.evaluate_sequence(sequence,6)

class ThresholdTest(BrickTest):
    def setUp(self):
        super(ThresholdTest, self).setUp()
        self.num_trials = 200
        self.tolerance = 0.05

    def process_results(self, results):
        graph_names = list(self.scaffold.graph.nodes.data('name'))
        spiked = False
        for row in results.itertuples():
            neuron_name = graph_names[int(row.neuron_number)][0]
            if "Thresh" in neuron_name:
                spiked = True
        return spiked

    def evaluate_scaffold(self, max_time):
        self.scaffold.lay_bricks()
        evaluations = 0.0
        hits = 0.0
        results = []
        for i in range(self.num_trials):
            evaluations += 1.0
            spiked = self.process_results(self.scaffold.evaluate(backend=BACKEND,max_runtime=max_time, backend_args=BACKEND_ARGS))
            if spiked:
                hits += 1
        return hits / evaluations

    def build_scaffold(self, output_coding, input_value, threshold, p_value, decay_value):
        self.scaffold.add_brick(BRICKS.Vector_Input(np.array([1]), coding='Raster', name='input1'), 'input' )
        self.scaffold.add_brick(BRICKS.Vector_Input(np.array([0]), coding='Raster', name='input2'), 'input' )
        self.scaffold.add_brick(BRICKS.Dot([input_value], name='ADotOperator'), (0,0)) #don't know why i need two vector inputs
        self.scaffold.add_brick(BRICKS.Threshold(threshold, 
                                                 p=p_value, 
                                                 decay=decay_value,
                                                 name='Thresh',
                                                 output_coding=output_coding),
                               (2,0), output=True)

    def test_current_no_spikes(self):
        self.build_scaffold('current', 1.0, 1, 1, 0)
        AssertValuesAreClose(0.0, self.evaluate_scaffold(5), self.tolerance)

    def test_current_always_spikes(self):
        self.build_scaffold('current', 1.01, 1, 1, 0)
        AssertValuesAreClose(1.0, self.evaluate_scaffold(5), self.tolerance)

    def test_current_sometimes_spikes(self):
        self.build_scaffold('current', 1.01, 1, 0.75, 0)
        AssertValuesAreClose(0.75, self.evaluate_scaffold(5), self.tolerance)

    def test_temporal_always_spikes_1(self):
        self.build_scaffold('temporal-L', 1, 0, 1, 0)
        AssertValuesAreClose(1.0, self.evaluate_scaffold(5), self.tolerance)

    def test_temporal_always_spikes_2(self):
        self.build_scaffold('temporal-L', 3, 2, 1, 0)
        AssertValuesAreClose(1.0, self.evaluate_scaffold(5), self.tolerance)

    def test_temporal_never_spikes_1(self):
        self.build_scaffold('temporal-L', 3, 3, 1, 0)
        AssertValuesAreClose(0.0, self.evaluate_scaffold(5), self.tolerance)

    def test_temporal_never_spikes_2(self):
        self.build_scaffold('temporal-L', 3, 4, 1, 0)
        AssertValuesAreClose(0.0, self.evaluate_scaffold(5), self.tolerance)

    def test_temporal_sometimes_spikes(self):
        self.build_scaffold('temporal-L', 3, 2, 0.23, 0)
        AssertValuesAreClose(0.23, self.evaluate_scaffold(5), self.tolerance)

class BFSTest(BrickTest):
    def setUp(self):
        super(BFSTest, self).setUp()
        self.graph = nx.DiGraph()

    def process_results(self, results):
        bfs_levels = {}
        bfs_names = list(self.scaffold.graph.nodes.data('name'))

        curr_level = 0
        curr_time = 0.0
        for row in results.sort_values("time").itertuples():
            neuron_name = bfs_names[int(row.neuron_number)][0]

            neuron_props = self.scaffold.graph.nodes[neuron_name]
            if 'is_vertex' in neuron_props:
                if row.time > curr_time:
                    curr_level += 1
                    curr_time = row.time
                vertex = neuron_props['index'][0]

                self.assertFalse(vertex in bfs_levels)

                bfs_levels[vertex] = curr_level

        for edge in self.graph.edges():
            u = edge[0]
            v = edge[1]
            u_level = bfs_levels[u]
            v_level = bfs_levels[v]
            self.assertTrue(abs(u_level - v_level) <= 1)

    def get_spike_input(self, search_key):
        spikes = [0] * search_key
        spikes[-1] = 1
        return spikes

    def evaluate_graph(self, search_key):
        bfs_input = BRICKS.Vector_Input(self.get_spike_input(search_key), coding='Raster', name='BFSInput')
        bfs_brick = BRICKS.Breadth_First_Search(self.graph, name="BFS")
        self.scaffold.add_brick(bfs_input, 'input')
        self.scaffold.add_brick(bfs_brick, output=True)
        self.evaluate_scaffold(len(self.graph.nodes) * 2)

    def test_random_gnp(self):
        self.graph = create_graph(20, 0.3, 3)
        self.evaluate_graph(1)

class SSSPTest(BrickTest):
    def setUp(self):
        super(SSSPTest, self).setUp()
        self.graph = nx.DiGraph()
        self.return_path = False

    def process_results(self, results):
        sssp_pred = {v:-1 for v in self.graph.nodes}
        sssp_table = {v:-1 for v in self.graph.nodes}
        sssp_start_time = 0.0

        sssp_names = list(self.scaffold.graph.nodes.data('name'))
        sssp_spikes = 0
        for row in results.itertuples():
            sssp_spikes += 1
            neuron_name = sssp_names[int(row.neuron_number)][0]

            neuron_props = self.scaffold.graph.nodes[neuron_name]
            if 'begin' in neuron_name:
                sssp_start_time = row.time
            elif 'is_vertex' in neuron_props:
                v = neuron_props['index'][0]
                sssp_table[v] = row.time
            if self.return_path:
                if 'is_edge_reference' in neuron_props:
                    u = neuron_props['from_vertex']
                    v = neuron_props['to_vertex']

                    sssp_pred[v] = u if u < sssp_pred[v] or sssp_pred[v] < 0 else sssp_pred[v]

        for v in sssp_table:
            if sssp_table[v] > -1:
                sssp_table[v] -= sssp_start_time
                sssp_table[v] /= 2.0

        for u,v in self.graph.edges():
            u_dist = sssp_table[u]
            v_dist = sssp_table[v]
            edge_weight = self.graph.get_edge_data(u,v)['weight']
            self.assertTrue(abs(u_dist - v_dist) <= edge_weight)

        if self.return_path:
            for u in sssp_pred:
                v = sssp_pred[u]
                if v > -1:
                    u_dist = sssp_table[u]
                    v_dist = sssp_table[v]

                    self.assertTrue(u_dist > -1)
                    self.assertTrue(v_dist > -1)

                    edge_weight = self.graph.get_edge_data(u,v)['weight']

                    self.assertTrue(abs(u_dist - v_dist) <= edge_weight)

    def get_spike_input(self, search_key):
        spikes = [0] * search_key
        spikes.append(1)
        return spikes

    def evaluate_graph(self, search_key):
        sssp_input = BRICKS.Vector_Input(self.get_spike_input(search_key), coding='Raster', name='SSSPInput')
        sssp_brick = BRICKS.Shortest_Path(self.graph, name="SSSP", return_path=self.return_path)
        self.scaffold.add_brick(sssp_input, 'input')
        self.scaffold.add_brick(sssp_brick, output=True)
        self.evaluate_scaffold(len(self.graph.edges) * 3 * 10)

    def test_race_condition(self): 
        self.return_path = False
        self.graph.add_edge(0,1,weight=1)
        self.graph.add_edge(1,2,weight=1)
        self.graph.add_edge(2,3,weight=1)
        self.graph.add_edge(3,4,weight=1)
        self.graph.add_edge(4,5,weight=1)
        self.graph.add_edge(5,6,weight=1)
        self.graph.add_edge(6,7,weight=1)
        self.graph.add_edge(7,8,weight=1)
        self.graph.add_edge(8,9,weight=1)
        self.graph.add_edge(0,10,weight=10)
        self.graph.add_edge(1,11,weight=9)
        self.graph.add_edge(2,12,weight=8)
        self.graph.add_edge(3,13,weight=7)
        self.graph.add_edge(4,14,weight=6)
        self.graph.add_edge(5,15,weight=5)
        self.graph.add_edge(6,16,weight=4)
        self.graph.add_edge(7,17,weight=3)
        self.graph.add_edge(8,18,weight=2)
        self.graph.add_edge(9,19,weight=1)
        self.evaluate_graph(0)

    def test_random_gnp_dist(self):
        self.graph = create_weighted_graph(20, 0.3, 3)
        self.evaluate_graph(1)

    def test_random_gnp_full(self):
        self.return_path = True
        self.graph = create_weighted_graph(20, 0.3, 3)
        self.evaluate_graph(1)
