import unittest
import numpy as np
import random

import fugu
import fugu.bricks as BRICKS
from fugu import Scaffold

def AssertValuesAreClose(value1, value2, tolerance = 0.0001):
    if abs(value1 - value2) > tolerance:
        raise AssertionError('Values {} and {} are not close'.format(value1, value2))

class UtilityBrickTests:
    backend = None
    backend_args = {}
    scale_factor = 0.5

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
        results = scaffold.evaluate(backend=self.backend, max_runtime=(sum(spike_times) + 5) * 2, backend_args=self.backend_args)

        answer = -1
        graph_names = list(scaffold.graph.nodes.data('name'))
        for row in results.itertuples():
            neuron_name = graph_names[int(row.neuron_number)][0]
            if 'Sum' in neuron_name:
                self.assertTrue(answer < 0)
                answer = self.scale_factor * row.time - 3
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

    def evaluate_register(self, spike_times, register_size):
        max_runtime = (2 ** (register_size + 2))
        scaffold = Scaffold()

        inputs = [[0]*max_runtime for i in range(2)]

        inputs[0][max_runtime - register_size - 1] = 1 # recall 'input'
        for spike in spike_times:
            inputs[1][spike] = 1

        scaffold.add_brick(BRICKS.Vector_Input(np.array(inputs), coding='Raster', name='input', time_dimension = True), 'input' )
        scaffold.add_brick(BRICKS.Register(register_size, name='register1'), output=True)

        scaffold.lay_bricks()

        result = scaffold.evaluate(backend=self.backend, max_runtime=max_runtime, backend_args=self.backend_args)

        value = 0
        graph_names = list(scaffold.graph.nodes.data('name'))
        for row in result.sort_values('time').itertuples():
            neuron_name = graph_names[int(row.neuron_number)][0]
            if 'output' in neuron_name:
                index = int(neuron_name.split('_')[1])
                value += (2 ** index)

        return value

    def test_register_1(self):
        answer = self.evaluate_register([5,20,45], 5)
        self.assertEqual(3, answer)

    def test_register_2(self):
        answer = self.evaluate_register([1 + i * 7 for i in range(10)], 7)
        self.assertEqual(10, answer)

    def test_register_3(self):
        answer = self.evaluate_register([i * 5 for i in range(19)], 5)
        self.assertEqual(19, answer)

    def test_register_4(self):
        answer = self.evaluate_register([1 + i * 8 for i in range(32)], 8)
        self.assertEqual(32, answer)

    def test_register_5(self):
        spike_times = [1]
        for i in range(25):
            spike_times.append(spike_times[-1] + random.randint(8, 12))
        answer = self.evaluate_register(spike_times, 8)
        self.assertEqual(26, answer)

class SnnUtilityTests(unittest.TestCase, UtilityBrickTests):
    @classmethod
    def setUpClass(self):
        self.backend = 'snn'

class DsUtilityTests(unittest.TestCase, UtilityBrickTests):
    @classmethod
    def setUpClass(self):
        self.backend = 'ds'

class PynnBrianUtilityTests(unittest.TestCase, UtilityBrickTests):
    @classmethod
    def setUpClass(self):
        self.backend = 'pynn'
        self.backend_args['backend'] = 'brian'

class PynnSpinnakerUtilityTests(unittest.TestCase, UtilityBrickTests):
    @classmethod
    def setUpClass(self):
        self.backend = 'pynn'
        self.backend_args['backend'] = 'spinnaker'
