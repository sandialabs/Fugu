import unittest
import numpy as np

import fugu
import fugu.bricks as BRICKS
from fugu.scaffold import Scaffold
from fugu.backends import ds_Backend, snn_Backend, pynn_Backend

from ..base import BrickTest


class InstantDecayTests(BrickTest):
    # Base class function
    def build_scaffold(self, input_times):
        scaffold = Scaffold()

        vector_brick = BRICKS.Vector_Input(
                                self.convert_input(input_times),
                                coding='Raster',
                                name='Input',
                                time_dimension=True,
                                )
        decay_brick = BRICKS.InstantDecay(len(input_times), name="InstantDecay")

        scaffold.add_brick(vector_brick, 'input')
        scaffold.add_brick(decay_brick, input_nodes=[(0, 0)], output=True)

        scaffold.lay_bricks()
        return scaffold

    def calculate_max_timesteps(self, input_values):
        max_time = 0
        for sequence in input_values:
            max_seq = max(sequence)
            if max_seq > max_time:
                max_time = max_seq

        return 2 * max_seq

    def check_spike_output(self, spikes, expected, scaffold):
        graph_names = list(scaffold.graph.nodes.data('name'))
        main_fired = False
        for row in spikes.itertuples():
            neuron_name = graph_names[int(row.neuron_number)][0]
            if "main" in neuron_name:
                main_fired = True

        self.assertEqual(expected, main_fired)

    def convert_input(self, spike_times):
        max_time = 0
        for spike_array in spike_times:
            last_spike = max(spike_array)
            if last_spike > max_time:
                max_time = last_spike

        converted_spikes = []
        for spike_array in spike_times:
            spikes = [0 for i in range(max_time + 1)]
            for spike in spike_array:
                spikes[spike] = 1
            converted_spikes.append(spikes)

        return converted_spikes

    # tests
    def test_fire(self):
        input_spikes = [[5, 10], [2, 10]]
        self.basic_test(input_spikes, True)

    def test_no_fire(self):
        input_spikes = [[1, 3, 5, 7, 9], [2, 4, 6, 8]]
        self.basic_test(input_spikes, False)


class SnnInstantDecayTests(InstantDecayTests, unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.backend = snn_Backend()


class DsInstantDecayTests(InstantDecayTests, unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.backend = ds_Backend()


class PynnSpinnakerInstantDecayTests(InstantDecayTests, unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.backend = pynn_Backend()
        self.backend_args['backend'] = 'spinnaker'


class PynnBrianInstantDecayTests(InstantDecayTests, unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.backend = pynn_Backend()
        self.backend_args['backend'] = 'brian'
