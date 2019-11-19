import unittest
import numpy as np
import random

import fugu
import fugu.bricks as BRICKS
from fugu import Scaffold


class UtilityBrickTests:
    backend = None
    backend_args = {}
    scale_factor = 0.5

    def evaluate_adder(self, spike_times, debug=False):
        scaffold = Scaffold()

        adder_brick = BRICKS.TemporalAdder(len(spike_times), name="Adder")

        max_time = max(spike_times)
        time_vector = [[0] * (2 * (max_time + 1)) for i in spike_times]
        time_vector[0][spike_times[0] * 2] = 1
        time_vector[1][spike_times[1] * 2] = 1

        vector_brick = BRICKS.Vector_Input(np.array(time_vector), coding='Raster', name='Input', time_dimension=True)

        scaffold.add_brick(vector_brick, 'input')
        scaffold.add_brick(adder_brick, output=True)

        scaffold.lay_bricks()
        results = scaffold.evaluate(backend=self.backend,
                                    max_runtime=(sum(spike_times) + 5) * 2,
                                    backend_args=self.backend_args,
                                    record_all=True)

        answer = -1
        graph_names = list(scaffold.graph.nodes.data('name'))
        for row in results.itertuples():
            neuron_name = graph_names[int(row.neuron_number)][0]
            if debug:
                print(neuron_name, row.time)
            if 'Sum' in neuron_name:
                self.assertTrue(answer < 0)
                answer = self.scale_factor * row.time - 3
        return answer

    def test_adder_1(self):
        result = self.evaluate_adder([10, 7])
        self.assertEqual(17, result)

    def test_adder_2(self):
        result = self.evaluate_adder([10, 8])
        self.assertEqual(18, result)

    def test_adder_3(self):
        result = self.evaluate_adder([6, 8])
        self.assertEqual(14, result)

    def test_adder_4(self):
        result = self.evaluate_adder([9, 9])
        self.assertEqual(18, result)

    def test_adder_5(self):
        result = self.evaluate_adder([1, 9])
        self.assertEqual(10, result)

    def evaluate_register(self, spike_times, register_size, initial_value=0.0, debug=False):
        max_runtime = (2 ** (register_size + 2))
        scaffold = Scaffold()

        inputs = [[0] * max_runtime for i in range(2)]

        inputs[0][max_runtime - register_size - 1] = 1  # recall 'input'
        for spike in spike_times:
            inputs[1][spike] = 1

        vector_brick = BRICKS.Vector_Input(np.array(inputs), coding='Raster', name='input', time_dimension=True)
        register_brick = BRICKS.Register(register_size, initial_value=initial_value, name='register1')
        scaffold.add_brick(vector_brick, 'input')
        scaffold.add_brick(register_brick, output=True)

        scaffold.lay_bricks()

        if debug:
            self.backend_args['verbose'] = True

        result = scaffold.evaluate(backend=self.backend, max_runtime=max_runtime, backend_args=self.backend_args)

        value = 0
        graph_names = list(scaffold.graph.nodes.data('name'))
        for row in result.sort_values('time').itertuples():
            neuron_name = graph_names[int(row.neuron_number)][0]
            if 'output' in neuron_name:
                index = int(neuron_name.split('_')[1])
                value += (2 ** index)

        return value

    @unittest.skip("Need to tweak pynn backend to support this")
    def test_register_1(self):
        answer = self.evaluate_register([5, 10, 25], 5)
        self.assertEqual(3, answer)

    @unittest.skip("Need to tweak pynn backend to support this")
    def test_register_2(self):
        answer = self.evaluate_register([1 + i * 7 for i in range(10)], 7)
        self.assertEqual(10, answer)

    @unittest.skip("Need to tweak pynn backend to support this")
    def test_register_3(self):
        answer = self.evaluate_register([i * 5 for i in range(19)], 5)
        self.assertEqual(19, answer)

    @unittest.skip("Need to tweak pynn backend to support this")
    def test_register_4(self):
        answer = self.evaluate_register([1 + i * 8 for i in range(32)], 8)
        self.assertEqual(32, answer)

    @unittest.skip("Need to tweak pynn backend to support this")
    def test_register_5(self):
        spike_times = [1]
        for i in range(25):
            spike_times.append(spike_times[-1] + random.randint(8, 12))
        answer = self.evaluate_register(spike_times, 8)
        self.assertEqual(26, answer)

    @unittest.skip("Need to tweak pynn backend to support this")
    def test_register_6(self):
        answer = self.evaluate_register([5, 20, 45], 8, initial_value=5)
        self.assertEqual(8, answer)

    @unittest.skip("Need to tweak pynn backend to support this")
    def test_register_7(self):
        answer = self.evaluate_register([], 6, initial_value=34)
        self.assertEqual(34, answer)


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
