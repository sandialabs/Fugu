import numpy as np

import fugu.bricks as BRICKS
from fugu.backends import snn_Backend
from fugu.scaffold import Scaffold

from ..brick_test import BrickTest


class TestSnnTemporalAdder(BrickTest):
    def setup_method(self):
        super().setup_method()
        self.backend = snn_Backend()
        self.scale_factor = 0.5

    # Base class overrides
    def build_scaffold(self, input_values):
        scaffold = Scaffold()

        converted_input = self.convert_input(input_values)
        vector_brick = BRICKS.Vector_Input(converted_input, coding="Raster", name="Input", time_dimension=True)

        temporal_adder_brick = BRICKS.TemporalAdder(len(input_values), name="TemporalAdder")

        scaffold.add_brick(vector_brick, "input")
        scaffold.add_brick(temporal_adder_brick, input_nodes=(0, 0), output=True)

        scaffold.lay_bricks()
        return scaffold

    def calculate_max_timesteps(self, input_values):
        return (sum(input_values) + 5) * 2

    def check_spike_output(self, spikes, expected, scaffold):
        answer = -1
        graph_names = list(scaffold.graph.nodes.data("name"))
        for row in spikes.itertuples():
            neuron_name = graph_names[int(row.neuron_number)][0]
            if "Sum" in neuron_name:
                assert answer < 0
                answer = self.scale_factor * row.time - 3

        assert expected == answer

    def convert_input(self, input_values):
        time_vector = [[0] * (2 * (max(input_values) + 1)) for i in input_values]
        time_vector[0][input_values[0] * 2] = 1
        time_vector[1][input_values[1] * 2] = 1
        return np.array(time_vector)

    # Tests
    def test_adder_1(self):
        self.basic_test([10, 7], 17)

    def test_adder_2(self):
        self.basic_test([10, 8], 18)

    def test_adder_3(self):
        self.basic_test([6, 8], 14)

    def test_adder_4(self):
        self.basic_test([9, 9], 18)

    def test_adder_5(self):
        self.basic_test([1, 9], 10)

    def teardown_method(self):
        super().teardown_method()
