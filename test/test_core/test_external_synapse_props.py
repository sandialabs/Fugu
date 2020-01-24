import unittest
import numpy as np

import fugu
import fugu.bricks as BRICKS
from fugu.scaffold import Scaffold
from fugu.backends import ds_Backend, snn_Backend, pynn_Backend

from ..base import BrickTest


class ChangeSynapseExternalPropertyTests(BrickTest):
    # Base class function
    def build_scaffold(self, input_values):
        scaffold = Scaffold()

        threshold_brick = BRICKS.Threshold(1.0,
                                           p=1.0,
                                           decay=0,
                                           name='Thresh',
                                           output_coding='temporal-L')
        vector = BRICKS.Vector_Input(np.array([1]), coding='Raster', name='input1')
        dot_brick = BRICKS.Dot([input_values[0]], name='ADotOperator')

        scaffold.add_brick(vector, 'input')
        scaffold.add_brick(dot_brick, input_nodes=[(0, 0)])
        scaffold.add_brick(threshold_brick, input_nodes=[(1, 0)], output=True)

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
    def test_change_external_synapose_properties(self):
        props = {}
        props = {}
        props['ADotOperator'] = {}
        props['ADotOperator']['weights'] = [2.1]

        self.run_property_test(
               [0.5],
               props,
               [
                 [],
                 [
                   ('Thresh', 1.0),
                 ],
               ],
               )


class SnnChangeSynapseExternalPropertyTests(ChangeSynapseExternalPropertyTests, unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.backend = snn_Backend()


class DsChangeSynapseExternalPropertyTests(ChangeSynapseExternalPropertyTests, unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.backend = ds_Backend()


class PynnSpinnakerChangeSynapseExternalPropertyTests(ChangeSynapseExternalPropertyTests, unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.backend = pynn_Backend()
        self.backend_args['backend'] = 'spinnaker'


class PynnBrianChangeSynapseExternalPropertyTests(ChangeSynapseExternalPropertyTests, unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.backend = pynn_Backend()
        self.backend_args['backend'] = 'brian'
