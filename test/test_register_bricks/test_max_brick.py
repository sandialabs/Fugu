import unittest
import numpy as np
import random

import fugu
import fugu.bricks as BRICKS
from fugu.scaffold import Scaffold
from fugu.backends import ds_Backend, snn_Backend, pynn_Backend

from ..base import BrickTest


class MaxBrickTests(BrickTest):
    # Base class overrides
    def build_scaffold(self, input_values):
        scaffold = Scaffold()

        converted_input = self.convert_input(input_values)

        max_brick = BRICKS.Max(name='Max')
        for i, value in enumerate(converted_input):
            vector_brick = BRICKS.Vector_Input(value, coding='Raster', name='Input{}'.format(i))
            scaffold.add_brick(vector_brick, 'input')
        scaffold.add_brick(max_brick, input_nodes=[(i, 0) for i in range(len(converted_input))], output=True)

        scaffold.lay_bricks()
        if self.debug:
            scaffold.summary(verbose=2)

        return scaffold

    def calculate_max_timesteps(self, input_values):
        return 30

    def check_spike_output(self, spikes, expected, scaffold):
        value = 0
        graph_names = list(scaffold.graph.nodes.data('name'))
        for row in spikes.sort_values('time').itertuples():
            brick_tag, neuron_name = graph_names[int(row.neuron_number)][0].split(":")
            if self.debug:
                print(neuron_name, row.time)
            if 'M' in neuron_name and 'Max' not in neuron_name:
                index = int(neuron_name.split('_')[1])
                value += (2 ** index)

        self.assertEqual(expected, value)

    def convert_input(self, input_values):
        vector_inputs = []
        max_value = max(input_values)
        for value in input_values:
            binary_string = "{:05b}".format(value)
            bit_array = np.array([int(b) for b in binary_string][::-1])
            vector_inputs.append(bit_array)

        return vector_inputs

    def test_max_single_value(self):
        values = [5]
        self.basic_test(values, max(values))

    def test_max_basic(self):
        values = [5, 3, 10, 2, 17, 25]
        self.basic_test(values, max(values))

    def test_max_tie(self):
        values = [13, 13, 1, 4, 11]
        self.basic_test(values, max(values))

    def test_max_all_tie(self):
        values = [23, 23, 23]
        self.basic_test(values, max(values))


class SnnMaxTests(MaxBrickTests, unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.backend = snn_Backend()


class DsMaxTests(MaxBrickTests, unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.backend = ds_Backend()


class PynnSpinnakerMaxTests(MaxBrickTests, unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.backend = pynn_Backend()
        self.backend_args['backend'] = 'spinnaker'


class PynnBrianMaxTests(MaxBrickTests, unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.backend = pynn_Backend()
        self.backend_args['backend'] = 'brian'
