import unittest
import numpy as np

import fugu
import fugu.bricks as BRICKS
from fugu.scaffold import Scaffold
from fugu.backends import ds_Backend, snn_Backend, pynn_Backend

from ..base import BrickTest


class VectorInputTests(BrickTest):
    # Base class function
    backend_args = {'record': 'all'}
    brick_name = "VectorInput"

    def build_scaffold(self, input_values):
        input_spikes, use_time = input_values
        scaffold = Scaffold()

        vector_brick = BRICKS.Vector_Input(
                                input_spikes,
                                coding='Raster',
                                name=self.brick_name,
                                time_dimension=use_time,
                                )

        scaffold.add_brick(vector_brick, input_nodes=['input'])

        scaffold.lay_bricks()
        return scaffold

    def check_spike_output(self, spikes, expected, scaffold):
        graph_names = list(scaffold.graph.nodes.data('name'))
        fire_list = [[] for n in expected]
        for row in spikes.itertuples():
            neuron_name = graph_names[int(row.neuron_number)][0]
            neuron_index = scaffold.graph.nodes[neuron_name]['index']
            if type(neuron_index) is not int:
                index = neuron_index[0]
                fire_list[index].append(int(row.time))

        self.assertEqual(expected, fire_list)

    # tests
    def test_spike_array(self):
        input_spikes = [1, 1, 0, 1]
        self.basic_test((input_spikes, False), [[0], [0], [], [0]])

    def test_time_spike_array(self):
        input_spikes = [[1, 1, 0], [0, 0, 1]]
        self.basic_test((input_spikes, True), [[0, 1], [2]])


class SnnVectorInputTests(VectorInputTests, unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.backend = snn_Backend()


class DsVectorInputTests(VectorInputTests, unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.backend = ds_Backend()


class PynnSpinnakerVectorInputTests(VectorInputTests, unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.backend = pynn_Backend()
        self.backend_args['backend'] = 'spinnaker'


class PynnBrianVectorInputTests(VectorInputTests, unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.backend = pynn_Backend()
        self.backend_args['backend'] = 'brian'
