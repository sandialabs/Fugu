import unittest
import numpy as np

import fugu
import fugu.bricks as BRICKS
from fugu.scaffold import Scaffold
from fugu.backends import ds_Backend, snn_Backend, pynn_Backend

from ..base import BrickTest


class ChangeSynapseInternalPropertyTests(BrickTest):
    # Base class function
    def build_scaffold(self, input_values):
        scaffold = Scaffold()

        scaffold.add_brick(BRICKS.SynapseProperties(weights=input_values, name='Test'), output=True)

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
    def test_change_internal_synapose_properties(self):
        props = {}
        props = {}
        props['Test'] = {}
        props['Test']['weights'] = [1.1, 1.1, 0.9, 2.1]

        self.run_property_test(
               [0.5, 0.1, 0.3, 0.4],
               props,
               [
                 [],
                 [
                   ('Test_0', 1.0),
                   ('Test_1', 1.0),
                   ('Test_3', 1.0),
                 ],
               ],
               )


class SnnChangeSynapseInternalPropertyTests(ChangeSynapseInternalPropertyTests, unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.backend = snn_Backend()


class DsChangeSynapseInternalPropertyTests(ChangeSynapseInternalPropertyTests, unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.backend = ds_Backend()


class PynnSpinnakerChangeSynapseInternalPropertyTests(ChangeSynapseInternalPropertyTests, unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.backend = pynn_Backend()
        self.backend_args['backend'] = 'spinnaker'


class PynnBrianChangeSynapseInternalPropertyTests(ChangeSynapseInternalPropertyTests, unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.backend = pynn_Backend()
        self.backend_args['backend'] = 'brian'
