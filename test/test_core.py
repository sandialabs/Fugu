import unittest
import numpy as np

import fugu
import fugu.bricks as BRICKS
from fugu import Scaffold


class CoreTests:
    backend = None
    backend_args = {}

    def process_spike_times(self, spike_times):
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
            

    def evaluate_instant_decay(self, input_times, expected, debug=False):
        scaffold = Scaffold()

        vector_brick = BRICKS.Vector_Input(
                                self.process_spike_times(input_times),
                                coding='Raster',
                                name='Input',
                                time_dimension=True,
                                )
        decay_brick = BRICKS.InstantDecay(len(input_times), name="InstantDecay")

        scaffold.add_brick(vector_brick, 'input')
        scaffold.add_brick(decay_brick, output=True)

        scaffold.lay_bricks()

        if debug:
            scaffold.summary(verbose=2)

        results = scaffold.evaluate(backend=self.backend, backend_args=self.backend_args, max_runtime=100)

        graph_names = list(scaffold.graph.nodes.data('name'))
        main_fired = False
        for row in results.itertuples():
            neuron_name = graph_names[int(row.neuron_number)][0]
            if debug:
                print(neuron_name, row.time)
            if "main" in neuron_name:
                main_fired = True

        self.assertEqual(expected, main_fired)

    def test_fire(self):
        input_spikes = [[5, 10], [2, 10]]
        self.evaluate_instant_decay(input_spikes, True)

    def test_no_fire(self):
        input_spikes = [[1,3,5,7,9], [2,4,6,8]]
        self.evaluate_instant_decay(input_spikes, False, debug=False)


class SnnCoreTests(unittest.TestCase, CoreTests):
    @classmethod
    def setUpClass(self):
        self.backend = 'snn'


class DsCoreTests(unittest.TestCase, CoreTests):
    @classmethod
    def setUpClass(self):
        self.backend = 'ds'


class PynnBrianCoreTests(unittest.TestCase, CoreTests):
    @classmethod
    def setUpClass(self):
        self.backend = 'pynn'
        self.backend_args['backend'] = 'brian'


class PynnSpinnakerCoreTests(unittest.TestCase, CoreTests):
    @classmethod
    def setUpClass(self):
        self.backend = 'pynn'
        self.backend_args['backend'] = 'spinnaker'
