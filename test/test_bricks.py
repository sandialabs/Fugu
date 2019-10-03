import abc
import sys
if sys.version_info >= (3, 4):
    ABC = abc.ABC
else:
    ABC = abc.ABCMeta('ABC', (), {'__slots__': ()})
from abc import abstractmethod

import unittest
import math

import fugu
import fugu.bricks as BRICKS
from fugu import Scaffold
#from fugu.bricks import Breadth_First_Search, Shortest_Path, Vector_Input

class BrickTest(unittest.TestCase, ABC):
    def setUp(self):
        super(BrickTest, self).setUp()
        self.scaffold = Scaffold()

    @abstractmethod
    def process_results(self, results):
        pass

    def evaluate_scaffold(self, max_time):
        self.scaffold.lay_bricks()
        results = self.scaffold.evaluate(backend='ds', max_runtime=max_time)
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

    def test_strictly_increasing(self):
        sequence = [1,2,3,4,5,6,7,8,9,10]
        self.scaffold.add_brick(BRICKS.Vector_Input(self.get_spike_input(sequence), coding='Raster', name='Input0', time_dimension=True), 'input')
        self.scaffold.add_brick(BRICKS.LIS(len(sequence), name="LIS"), output=True)
        answer = self.evaluate_scaffold(100)
        self.assertEqual(len(sequence), answer)

    def test_strictly_decreasing(self):
        sequence = [10,10,5,5,1]
        self.scaffold.add_brick(BRICKS.Vector_Input(self.get_spike_input(sequence), coding='Raster', name='Input0', time_dimension=True), 'input')
        self.scaffold.add_brick(BRICKS.LIS(len(sequence), name="LIS"), output=True)
        answer = self.evaluate_scaffold(100)
        self.assertEqual(1, answer)

    def test_general1(self):
        sequence = [5, 10, 2, 3, 4]
        self.scaffold.add_brick(BRICKS.Vector_Input(self.get_spike_input(sequence), coding='Raster', name='Input0', time_dimension=True), 'input')
        self.scaffold.add_brick(BRICKS.LIS(len(sequence), name="LIS"), output=True)
        answer = self.evaluate_scaffold(100)
        self.assertEqual(3, answer)

    def test_general2(self):
        sequence = [1,4,8,6,2,7,19,13,14]
        self.scaffold.add_brick(BRICKS.Vector_Input(self.get_spike_input(sequence), coding='Raster', name='Input0', time_dimension=True), 'input')
        self.scaffold.add_brick(BRICKS.LIS(len(sequence), name="LIS"), output=True)
        answer = self.evaluate_scaffold(100)
        self.assertEqual(6, answer)
