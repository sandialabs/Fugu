import unittest
import numpy as np

import fugu
import fugu.bricks as BRICKS
from fugu import Scaffold

class AppBrickTests:
    backend = None
    backend_args = {}

    def get_lis_input(self, sequence):
        num_in_sequence = len(sequence)
        max_time = max(sequence)
        spike_times = [[0] * (max_time + 1) for i in range(num_in_sequence)]
        for i, time in enumerate(sequence):
            spike_times[i][time] = 1
        return spike_times

    def evaluate_lis_sequence(self, sequence, expected, debug=False):
        scaffold = Scaffold()

        vector_brick = BRICKS.Vector_Input(
                                self.get_lis_input(sequence),
                                coding='Raster',
                                name='Input0',
                                time_dimension=True,
                                )
        lis_brick = BRICKS.LIS(len(sequence), name="LIS")

        scaffold.add_brick(vector_brick, 'input')
        scaffold.add_brick(lis_brick, output=True)

        scaffold.lay_bricks()

        if debug:
            scaffold.summary(verbose=2)

        results = scaffold.evaluate(backend=self.backend, backend_args=self.backend_args, max_runtime=100)

        graph_names = list(scaffold.graph.nodes.data('name'))
        answer = 0
        for row in results.itertuples():
            neuron_name = graph_names[int(row.neuron_number)][0]
            if debug:
                print(neuron_name, row.time)
            if "Main" in neuron_name:
                level = int(neuron_name.split("_")[1])
                if level > answer:
                    answer = level

        self.assertEqual(expected, answer)

    def test_lis_strictly_increasing(self):
        sequence = [1,2,3,4,5,6,7,8,9,10]
        self.evaluate_lis_sequence(sequence,len(sequence))

    def test_lis_strictly_decreasing(self):
        sequence = [10,10,5,5,1]
        self.evaluate_lis_sequence(sequence,1)

    def test_lis_general1(self):
        sequence = [5, 10, 2, 3, 4]
        self.evaluate_lis_sequence(sequence,3)

    def test_lis_general2(self):
        sequence = [1,4,8,6,2,7,19,13,14]
        self.evaluate_lis_sequence(sequence,6)

class SnnAppTests(unittest.TestCase, AppBrickTests):
    @classmethod
    def setUpClass(self):
        self.backend = 'snn'

class DsAppTests(unittest.TestCase, AppBrickTests):
    @classmethod
    def setUpClass(self):
        self.backend = 'ds'

class PynnBrianAppTests(unittest.TestCase, AppBrickTests):
    @classmethod
    def setUpClass(self):
        self.backend = 'pynn'
        self.backend_args['backend'] = 'brian'

class PynnSpinnakerAppTests(unittest.TestCase, AppBrickTests):
    @classmethod
    def setUpClass(self):
        self.backend = 'pynn'
        self.backend_args['backend'] = 'spinnaker'
