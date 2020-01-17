import unittest
import numpy as np

import fugu
import fugu.bricks as BRICKS
from fugu import Scaffold
from fugu import ds_Backend, snn_Backend, pynn_Backend

from ..base import BrickTest


class ChangeNeuronPropertyTests(BrickTest):
    # Base class function
    def build_scaffold(self, input_values):
        scaffold = Scaffold()
        spike_times, old_threshold = input_values

        vector_1 = BRICKS.Vector_Input(spike_times, coding='Raster', name='input1')
        dot_brick = BRICKS.Dot([1.0 for t in spike_times], name='Dot')
        thresh = BRICKS.Threshold(old_threshold, name='Thresh', output_coding='temporal-L')

        scaffold.add_brick(vector_1, 'input')
        scaffold.add_brick(dot_brick, input_nodes=(0, 0))
        scaffold.add_brick(thresh, input_nodes=(1, 0), output=True)

        scaffold.lay_bricks()
        return scaffold

    def check_spike_output(self, spikes, expected, scaffold):
        graph_names = list(scaffold.graph.nodes.data('name'))
        before_spikes, after_spikes = spikes
        before_expected, after_expected = expected

        processed = set()
        for row in before_spikes.itertuples():
            neuron_name = graph_names[int(row.neuron_number)][0]
            processed.add((neuron_name, row.time))
        for row in after_spikes.itertuples():
            neuron_name = graph_names[int(row.neuron_number)][0]
            processed.add((neuron_name, row.time))

        for entry in before_expected:
            self.assertTrue(entry in processed)
            processed.remove(entry)
        for entry in after_expected:
            self.assertTrue(entry in processed)
            processed.remove(entry)

        self.assertTrue(len(processed) == 0)

    # tests
    def test_change_neuron_properties(self):
        params = {}
        params['Thresh'] = {}
        params['Thresh']['threshold'] = 1.0
        self.run_parameter_test([[0, 1, 0, 1], 3.0], params, [[], [('Thresh', 1.0)]])


class SnnChangeNeuronPropertyTests(ChangeNeuronPropertyTests, unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.backend = snn_Backend()


class DsChangeNeuronPropertyTests(ChangeNeuronPropertyTests, unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.backend = ds_Backend()


class PynnSpinnakerChangeNeuronPropertyTests(ChangeNeuronPropertyTests, unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.backend = pynn_Backend()
        self.backend_args['backend'] = 'spinnaker'


class PynnBrianChangeNeuronPropertyTests(ChangeNeuronPropertyTests, unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.backend = pynn_Backend()
        self.backend_args['backend'] = 'brian'
