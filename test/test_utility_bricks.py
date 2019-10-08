import unittest
import numpy as np

import fugu
import fugu.bricks as BRICKS
from fugu import Scaffold

def AssertValuesAreClose(value1, value2, tolerance = 0.0001):
    if abs(value1 - value2) > tolerance:
        raise AssertionError('Values {} and {} are not close'.format(value1, value2))

class UtilityBrickTests:
    backend = None
    backend_args = {}

    def evaluate_adder(self, spike_times):
        scaffold = Scaffold()

        adder_brick = BRICKS.TemporalAdder(len(spike_times), name="Adder")
        max_time = max(spike_times)
        time_vector = [[0] * (2 * (max_time + 1)) for i in spike_times]
        time_vector[0][spike_times[0] * 2] = 1
        time_vector[1][spike_times[1] * 2] = 1
        scaffold.add_brick(BRICKS.Vector_Input(np.array(time_vector), coding='Raster', name='Input', time_dimension=True), 'input')
        scaffold.add_brick(adder_brick, output=True)

        scaffold.lay_bricks()
        results = scaffold.evaluate(backend=self.backend, max_runtime=(sum(spike_times) + 4) * 2, record_all=True,  backend_args=self.backend_args)

        answer = -1
        graph_names = list(scaffold.graph.nodes.data('name'))
        for row in results.itertuples():
            neuron_name = graph_names[int(row.neuron_number)][0]
            #print(neuron_name, row.time)
            if 'Sum' in neuron_name:
                self.assertTrue(answer < 0)
                answer = 0.5 * row.time - 3
        return answer

    def test_adder_1(self):
        result = self.evaluate_adder([10,7])
        self.assertEqual(17, result)

    def test_adder_2(self):
        result = self.evaluate_adder([10,8])
        self.assertEqual(18, result)

    def test_adder_3(self):
        result = self.evaluate_adder([6,8])
        self.assertEqual(14, result)

    def test_adder_4(self):
        result = self.evaluate_adder([9,9])
        self.assertEqual(18, result)

    def test_adder_5(self):
        result = self.evaluate_adder([1,9])
        self.assertEqual(10, result)

class SnnBackendUtilityTests(unittest.TestCase, UtilityBrickTests):
    @classmethod
    def setUpClass(self):
        self.backend = 'snn'

class DsBackendUtilityTests(unittest.TestCase, UtilityBrickTests):
    @classmethod
    def setUpClass(self):
        self.backend = 'ds'

class PynnBrianBackendUtilityTests(unittest.TestCase, UtilityBrickTests):
    @classmethod
    def setUpClass(self):
        self.backend = 'pynn'
        self.backend_args['backend'] = 'brian'

class PynnSpinnakerBackendUtilityTests(unittest.TestCase, UtilityBrickTests):
    @classmethod
    def setUpClass(self):
        self.backend = 'pynn'
        self.backend_args['backend'] = 'spinnaker'
