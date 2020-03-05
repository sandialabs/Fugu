import unittest
import numpy as np

import fugu
import fugu.bricks as BRICKS
from fugu.scaffold import Scaffold
from fugu.backends import ds_Backend, snn_Backend, pynn_Backend

from ..base import BrickTest


class AdditionBrickTests(BrickTest):
    scale_factor = 0.5

    # Base class overrides
    def build_scaffold(self, input_values):
        scaffold = Scaffold()

        converted_input = self.convert_input(input_values)
        vector_1 = BRICKS.Vector_Input(converted_input[0], coding='Raster', name='Input1')
        vector_2 = BRICKS.Vector_Input(converted_input[1], coding='Raster', name='Input2')

        addition_brick = BRICKS.Addition(register_size=5, name="Addition")

        scaffold.add_brick(vector_1, 'input')
        scaffold.add_brick(vector_2, 'input')
        scaffold.add_brick(addition_brick, input_nodes=[(0, 0), (1, 0)], output=True)

        scaffold.lay_bricks()
        return scaffold

    def calculate_max_timesteps(self, input_values):
        return 4

    def check_spike_output(self, spikes, expected, scaffold):
        answer = 0
        graph_names = list(scaffold.graph.nodes.data('name'))
        for row in spikes.itertuples():
            node_name = graph_names[int(row.neuron_number)][0]
            brick_tag, neuron_name = node_name.split(":")
            if self.debug:
                print(node_name, row.time)
            if 'S' in neuron_name and 'Addition' in brick_tag:
                bit_position = scaffold.graph.nodes[node_name]['bit_position']
                answer += 2 ** bit_position

        self.assertEqual(expected, answer)

    def convert_input(self, input_values):
        converted_inputs = []
        for value in input_values:
            binary = [int(b) for b in "{:0b}".format(value)[::-1]]
            converted_inputs.append(np.array(binary))

        return converted_inputs

    # Tests
    def test_addition_1(self):
        self.basic_test([10, 7], 17)

    def test_addition_2(self):
        self.basic_test([10, 8], 18)

    def test_addition_3(self):
        self.basic_test([6, 8], 14)

    def test_addition_4(self):
        self.basic_test([9, 9], 18)

    def test_addition_5(self):
        self.basic_test([1, 9], 10)

    def test_addition_overflow(self):
        self.basic_test([10, 25], 3)


class SnnAdditionTests(AdditionBrickTests, unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.backend = snn_Backend()


class DsAdditionTests(AdditionBrickTests, unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.backend = ds_Backend()


class PynnSpinnakerAdditionTests(AdditionBrickTests, unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.backend = pynn_Backend()
        self.backend_args['backend'] = 'spinnaker'


class PynnBrianAdditionTests(AdditionBrickTests, unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.backend = pynn_Backend()
        self.backend_args['backend'] = 'brian'
