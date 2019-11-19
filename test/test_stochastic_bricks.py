import unittest
import numpy as np

import fugu
import fugu.bricks as BRICKS
from fugu import Scaffold

from utilities import AssertValuesAreClose


class StochasticBrickTests:
    num_trials = 100
    tolerance = 0.15
    backend = None
    backend_args = {}

    def evaluate_thresh_params(self, output_coding, input_value, threshold, p_value, decay_value):
        scaffold = Scaffold()

        threshold_brick = BRICKS.Threshold(threshold,
                                           p=p_value,
                                           decay=decay_value,
                                           name='Thresh',
                                           output_coding=output_coding)
        vector_1 = BRICKS.Vector_Input(np.array([0]), coding='Raster', name='input1')
        vector_2 = BRICKS.Vector_Input(np.array([1]), coding='Raster', name='input2')
        dot_brick = BRICKS.Dot([input_value], name='ADotOperator')

        scaffold.add_brick(vector_1, 'input')
        scaffold.add_brick(vector_2, 'input')
        scaffold.add_brick(dot_brick)
        scaffold.add_brick(threshold_brick, (2, 0), output=True)

        scaffold.lay_bricks()

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

    def test_thresh_current_no_spikes(self):
        result = self.evaluate_thresh_params('current', 1.0, 1, 1, 0)
        AssertValuesAreClose(0.0, result, self.tolerance)

    def test_thresh_current_always_spikes(self):
        result = self.evaluate_thresh_params('current', 1.01, 1, 1, 0)
        AssertValuesAreClose(1.0, result, self.tolerance)

    def test_thresh_current_sometimes_spikes(self):
        result = self.evaluate_thresh_params('current', 1.01, 1, 0.75, 0)
        AssertValuesAreClose(0.75, result, self.tolerance)

    def test_thresh_temporal_always_spikes_1(self):
        result = self.evaluate_thresh_params('temporal-L', 1, 0, 1, 0)
        AssertValuesAreClose(1.0, result, self.tolerance)

    def test_thresh_temporal_always_spikes_2(self):
        result = self.evaluate_thresh_params('temporal-L', 3, 2, 1, 0)
        AssertValuesAreClose(1.0, result, self.tolerance)

    def test_thresh_temporal_never_spikes_1(self):
        result = self.evaluate_thresh_params('temporal-L', 3, 3, 1, 0)
        AssertValuesAreClose(0.0, result, self.tolerance)

    def test_thresh_temporal_never_spikes_2(self):
        result = self.evaluate_thresh_params('temporal-L', 3, 4, 1, 0)
        AssertValuesAreClose(0.0, result, self.tolerance)

    def test_thresh_temporal_sometimes_spikes(self):
        result = self.evaluate_thresh_params('temporal-L', 3, 2, 0.13, 0)
        AssertValuesAreClose(0.13, result, self.tolerance)


class SnnStochasticTests(unittest.TestCase, StochasticBrickTests):
    @classmethod
    def setUpClass(self):
        self.backend = 'snn'
        self.tolerance = .30


class DsStochasticTests(unittest.TestCase, StochasticBrickTests):
    @classmethod
    def setUpClass(self):
        self.backend = 'ds'
