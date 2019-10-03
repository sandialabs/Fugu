import abc
import sys
if sys.version_info >= (3, 4):
    ABC = abc.ABC
else:
    ABC = abc.ABCMeta('ABC', (), {'__slots__': ()})
from abc import abstractmethod

import unittest
import math
import numpy as np

import fugu
import fugu.bricks as BRICKS
from fugu import Scaffold
#from fugu.bricks import Breadth_First_Search, Shortest_Path, Vector_Input

def AssertValuesAreClose(value1, value2, tolerance = 0.0001):
    if not math.isclose(value1, value2, abs_tol=tolerance):
        raise AssertionError('Values {} and {} are not close'.format(value1, value2))

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
            spiked = self.process_results(self.scaffold.evaluate(backend='ds',max_runtime=max_time, record_all=True))
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
        self.build_scaffold('temporal-L', 3, 2, 0.65, 0)
        AssertValuesAreClose(0.65, self.evaluate_scaffold(5), self.tolerance)
