import unittest
import numpy as np

import fugu
import fugu.bricks as BRICKS
from fugu import Scaffold
from fugu import ds_Backend, snn_Backend, pynn_Backend

from ..base import BrickTest


class AdderBrickTests(BrickTest):
    scale_factor = 0.5

    # Base class overrides
    def build_scaffold(self, input_values):
        scaffold = Scaffold()

        converted_input = self.convert_input(input_values)
        vector_brick = BRICKS.Vector_Input(converted_input, coding='Raster', name='Input', time_dimension=True)

        adder_brick = BRICKS.TemporalAdder(len(input_values), name="Adder")

        scaffold.add_brick(vector_brick, 'input')
        scaffold.add_brick(adder_brick, input_nodes=(0, 0), output=True)

        scaffold.lay_bricks()
        return scaffold

    def calculate_max_timesteps(self, input_values):
        return (sum(input_values) + 5) * 2

    def check_spike_output(self, spikes, expected, scaffold):
        answer = -1
        graph_names = list(scaffold.graph.nodes.data('name'))
        for row in spikes.itertuples():
            neuron_name = graph_names[int(row.neuron_number)][0]
            if 'Sum' in neuron_name:
                self.assertTrue(answer < 0)
                answer = self.scale_factor * row.time - 3

        self.assertEqual(expected, answer)

    def convert_input(self, input_values):
        time_vector = [[0] * (2 * (max(input_values) + 1)) for i in input_values]
        time_vector[0][input_values[0] * 2] = 1
        time_vector[1][input_values[1] * 2] = 1
        return np.array(time_vector)

    # Tests
    def test_adder_1(self):
        self.basic_test([10, 7], 17)

    def test_adder_2(self):
        self.basic_test([10, 8], 18)

    def test_adder_3(self):
        self.basic_test([6, 8], 14)

    def test_adder_4(self):
        self.basic_test([9, 9], 18)

    def test_adder_5(self):
        self.basic_test([1, 9], 10)


class SnnAdderTests(AdderBrickTests, unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.backend = snn_Backend()


class DsAdderTests(AdderBrickTests, unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.backend = ds_Backend()


class PynnSpinnakerAdderTests(AdderBrickTests, unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.backend = pynn_Backend()
        self.backend_args['backend'] = 'spinnaker'


class PynnBrianAdderTests(AdderBrickTests, unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.backend = pynn_Backend()
        self.backend_args['backend'] = 'brian'
