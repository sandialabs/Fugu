import unittest
import numpy as np

import fugu
import fugu.bricks as BRICKS
from fugu.scaffold import Scaffold
from fugu.backends import ds_Backend, snn_Backend, pynn_Backend

from ..base import BrickTest


class CompoundBrickTests(BrickTest):
    # Base class function
    def build_scaffold(self, input_values):
        scaffold = Scaffold()

        converted_inputs = self.convert_input(input_values)

        num_of_sets = len(input_values)
        num_of_inputs = 0
        set_sizes = []
        for input_set in converted_inputs:
            set_sizes.append(len(input_set))
            for input_value in input_set:
                vector = BRICKS.Vector_Input(
                    input_value,
                    coding='Raster',
                    name="Input{}".format(num_of_inputs))
                scaffold.add_brick(vector, 'input')
                num_of_inputs += 1
        sum_of_maxes = BRICKS.SumOfMaxes(set_sizes, name="SumOfMaxes")
        scaffold.add_brick(sum_of_maxes,
                           input_nodes=[(i, 0) for i in range(num_of_inputs)],
                           output=True)

        scaffold.lay_bricks()

        return scaffold

    def check_spike_output(self, spikes, expected, scaffold):
        graph_names = list(scaffold.graph.nodes.data('name'))
        answer = 0

        for row in sorted(spikes.itertuples(), key=lambda x: x[-1]):
            node_name = graph_names[int(row.neuron_number)][0]
            tags = node_name.split(":")
            brick_tag = tags[0]
            neuron_name = tags[-1]
            if self.debug:
                print(node_name, row.time)
            if 'S' in neuron_name and 'SumOfMaxes' in brick_tag:
                bit_position = scaffold.graph.nodes[node_name]['bit_position']
                answer += 2**bit_position

        self.assertEqual(expected, answer)

    def calculate_max_timesteps(self, input_values):
        return 130

    def convert_input(self, input_values):
        converted_values = []
        for input_set in input_values:
            converted_set = []
            for value in input_set:
                binary = [int(b) for b in "{:05b}".format(value)[::-1]]
                converted_set.append(np.array(binary))
            converted_values.append(converted_set)

        return converted_values

    # tests
    def test_equal_maxes(self):
        inputs = [[5, 17, 13, 4], [17, 16, 15, 3, 2]]
        self.basic_test(inputs, sum([max(i) for i in inputs]))

    def test_same_sets(self):
        inputs = [[5, 4], [5, 4]]
        self.basic_test(inputs, sum([max(i) for i in inputs]))

    def test_different_maxes(self):
        inputs = [[5, 4], [3, 5, 4, 12]]
        self.basic_test(inputs, sum([max(i) for i in inputs]))


class SnnCompoundBrickTests(CompoundBrickTests, unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.backend = snn_Backend()


class DsCompoundBrickTests(CompoundBrickTests, unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.backend = ds_Backend()


class PynnSpinnakerCompoundBrickTests(CompoundBrickTests, unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.backend = pynn_Backend()
        self.backend_args['backend'] = 'spinnaker'


class PynnBrianCompoundBrickTests(CompoundBrickTests, unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.backend = pynn_Backend()
        self.backend_args['backend'] = 'brian'
