import unittest
import numpy as np

import fugu
import fugu.bricks as BRICKS
from fugu.scaffold import Scaffold
from fugu.backends import snn_Backend

from ..base import BrickTest


class SubtractionBrickTests(BrickTest):
    # Base class overrides
    def build_scaffold(self, input_values):
        scaffold = Scaffold()

        converted_input = self.convert_input(input_values)
        vector_1 = BRICKS.Vector_Input(converted_input[0], coding='Raster', name='Input1')
        vector_2 = BRICKS.Vector_Input(converted_input[1], coding='Raster', name='Input2')

        subtraction_brick = BRICKS.Subtraction(name="Subtraction")

        scaffold.add_brick(vector_1, 'input')
        scaffold.add_brick(vector_2, 'input')
        scaffold.add_brick(subtraction_brick, input_nodes=[(0, 0), (1, 0)], output=True)

        scaffold.lay_bricks()
        return scaffold

    def calculate_max_timesteps(self, input_values):
        return 10

    def check_spike_output(self, spikes, expected, scaffold):
        answer = 0
        graph_names = list(scaffold.graph.nodes.data('name'))
        for row in spikes.itertuples():
            node_name = graph_names[int(row.neuron_number)][0]
            tags = node_name.split(":")
            brick_tag = tags[0]
            neuron_name = tags[-1]
            if self.debug:
                print(node_name, row.time)
            if 'S' in neuron_name and 'Subtraction' in brick_tag:
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
    def test_subtraction_1(self):
        self.basic_test([10, 7], 3)

    def test_subtraction_2(self):
        self.basic_test([10, 8], 2)

    def test_subtraction_3(self):
        self.basic_test([9, 9], 0)

    def test_subtraction_4(self):
        self.basic_test([23, 2], 21)


class SnnSubtractionTests(SubtractionBrickTests, unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.backend = snn_Backend()
