import unittest
import numpy as np
import random

import fugu
import fugu.bricks as BRICKS
from fugu import Scaffold

from ..base import BrickTest


class RegisterBrickTests(BrickTest):
    # Base class overrides
    def build_scaffold(self, input_values):
        spike_times, register_size, initial_value = input_values

        scaffold = Scaffold()

        converted_input = self.convert_input(input_values)

        vector_brick = BRICKS.Vector_Input(converted_input, coding='Raster', name='input', time_dimension=True)
        register_brick = BRICKS.Register(register_size, initial_value=initial_value, name='register1')

        scaffold.add_brick(vector_brick, 'input')
        scaffold.add_brick(register_brick, output=True)

        scaffold.lay_bricks()

    def calculate_max_timesteps(self, input_values):
        _, register_size, _ = input_values
        return (2 ** (register_size + 2))

    def check_spike_output(self, spikes, expected, scaffold):
        value = 0
        graph_names = list(scaffold.graph.nodes.data('name'))
        for row in result.sort_values('time').itertuples():
            neuron_name = graph_names[int(row.neuron_number)][0]
            if 'output' in neuron_name:
                index = int(neuron_name.split('_')[1])
                value += (2 ** index)

        self.assertEqual(expected, value)

    def convert_input(self, input_values):
        spike_times, register_size, _ = input_values
        max_runtime = (2 ** (register_size + 2))

        inputs = [[0] * max_runtime for i in range(2)]

        inputs[0][max_runtime - register_size - 1] = 1  # recall 'input'
        for spike in spike_times:
            inputs[1][spike] = 1

        return np.array(inputs)

    def test_register_basic(self):
        self.basic_test([[5, 10, 25], 5, 0], 3)

    def test_register_uniform_intervals_1(self):
        self.basic_test([[1 + i * 7 for i in range(10)], 7, 0], 10)

    def test_register_uniform_intervals_2(self):
        self.basic_test([[i * 5 for i in range(19)], 5, 0], 19)

    def test_register_uniform_intervals_3(self):
        self.basic_test([[1 + i * 8 for i in range(32)], 8, 0], 32)

    def test_register_random_intervals(self):
        spike_times = [1]
        for i in range(25):
            spike_times.append(spike_times[-1] + random.randint(8, 12))
        self.basic_test([spike_times, 8, 0], 26)

    def test_register_add_to_initial_value(self):
        self.basic_test([[5, 20, 45], 8, 5], 8)

    def test_register_only_initial_value(self):
        self.basic_test([[], 6, 34], 34)


class SnnRegisterTests(unittest.TestCase, RegisterBrickTests):
    @classmethod
    def setUpClass(self):
        self.backend = 'snn'


class DsRegisterTests(unittest.TestCase, RegisterBrickTests):
    @classmethod
    def setUpClass(self):
        self.backend = 'ds'


class PynnBrianRegisterTests(unittest.TestCase, RegisterBrickTests):
    @classmethod
    def setUpClass(self):
        self.backend = 'pynn'
        self.backend_args['backend'] = 'brian'


class PynnSpinnakerRegisterTests(unittest.TestCase, RegisterBrickTests):
    @classmethod
    def setUpClass(self):
        self.backend = 'pynn'
        self.backend_args['backend'] = 'spinnaker'
