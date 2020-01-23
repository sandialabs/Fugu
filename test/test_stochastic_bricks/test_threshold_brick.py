import unittest
import numpy as np

import fugu
import fugu.bricks as BRICKS
from fugu.scaffold import Scaffold
from fugu.backends import ds_Backend, snn_Backend, pynn_Backend

from ..base import BrickTest
from ..utilities import AssertValuesAreClose


class ThresholdBrickTests(BrickTest):
    num_trials = 100
    tolerance = 0.15

    def setUp(self):
        self.hits = 0

    # Base class function
    def build_scaffold(self, input_values):
        scaffold = Scaffold()
        coding, weight, threshold, p_value, decay_value = input_values

        threshold_brick = BRICKS.Threshold(threshold,
                                           p=p_value,
                                           decay=decay_value,
                                           name='Thresh',
                                           output_coding=coding)
        vector = BRICKS.Vector_Input(np.array([1]), coding='Raster', name='input1')
        dot_brick = BRICKS.Dot([weight], name='ADotOperator')

        scaffold.add_brick(vector, 'input')
        scaffold.add_brick(dot_brick, input_nodes=[(0, 0)])
        scaffold.add_brick(threshold_brick, input_nodes=[(1, 0)], output=True)

        scaffold.lay_bricks()
        return scaffold

    def update_hit_count(self, spikes, scaffold):
        graph_names = list(scaffold.graph.nodes.data('name'))
        spiked = False
        for row in spikes.itertuples():
            neuron_name = graph_names[int(row.neuron_number)][0]
            if "Thresh" in neuron_name:
                spiked = True
        if spiked:
            self.hits += 1

    def evaluate_thresh_params(self, input_value, threshold, p_value, decay_value):

        evaluations = 0.0
        hits = 0.0
        results = []
        for i in range(self.num_trials):
            evaluations += 1.0
            results = scaffold.evaluate(backend=self.backend,
                                        max_runtime=5,
                                        backend_args=self.backend_args,
                                        record_all=True)
            graph_names = list(scaffold.graph.nodes.data('name'))
            spiked = False
            for row in results.itertuples():
                neuron_name = graph_names[int(row.neuron_number)][0]
                if "Thresh" in neuron_name:
                    spiked = True
            if spiked:
                hits += 1
        return hits / evaluations

    def run_iterations(self, expected, scaffold):
        self.backend.compile(scaffold)
        for i in range(self.num_trials):
            self.backend.reset()
            spikes = self.backend.run(5)
            self.update_hit_count(spikes, scaffold)

        AssertValuesAreClose(expected, self.hits / float(self.num_trials), self.tolerance)

    def test_thresh_current_no_spikes(self):
        scaffold = self.build_scaffold(['current', 1.0, 1, 1, 0])
        self.run_iterations(0.0, scaffold)

    def test_thresh_current_always_spikes(self):
        scaffold = self.build_scaffold(['current', 1.01, 1, 1, 0])
        self.run_iterations(1.0, scaffold)

    def test_thresh_current_sometimes_spikes(self):
        scaffold = self.build_scaffold(['current', 1.01, 1, 0.75, 0])
        self.run_iterations(0.75, scaffold)

    def test_thresh_temporal_always_spikes_1(self):
        scaffold = self.build_scaffold(['temporal-L', 1, 0, 1, 0])
        self.run_iterations(1.0, scaffold)

    def test_thresh_temporal_always_spikes_2(self):
        scaffold = self.build_scaffold(['temporal-L', 3, 2, 1, 0])
        self.run_iterations(1.0, scaffold)

    def test_thresh_temporal_never_spikes_1(self):
        scaffold = self.build_scaffold(['temporal-L', 3, 3, 1, 0])
        self.run_iterations(0.0, scaffold)

    def test_thresh_temporal_never_spikes_2(self):
        scaffold = self.build_scaffold(['temporal-L', 3, 4, 1, 0])
        self.run_iterations(0.0, scaffold)

    def test_thresh_temporal_sometimes_spikes(self):
        scaffold = self.build_scaffold(['temporal-L', 3, 2, 0.13, 0])
        self.run_iterations(0.13, scaffold)


class SnnThresholdTests(ThresholdBrickTests, unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.backend = snn_Backend()
        self.tolerance = .30


class DsThresholdTests(ThresholdBrickTests, unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.backend = ds_Backend()
