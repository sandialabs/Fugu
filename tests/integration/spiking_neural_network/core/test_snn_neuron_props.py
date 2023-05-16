import pytest
from brick_test import BrickTest

import fugu.bricks as BRICKS
from fugu.backends import snn_Backend
from fugu.scaffold import Scaffold


class TestSnnChangeNeuronProperty(BrickTest):
    def setup_method(self):
        super().setup_method()
        self.backend = snn_Backend()

    # Base class function
    def build_scaffold(self, input_values):
        scaffold = Scaffold()
        spike_times, old_threshold = input_values

        vector_1 = BRICKS.Vector_Input(spike_times, coding="Raster", name="input1")
        dot_brick = BRICKS.Dot([1.0 for t in spike_times], name="Dot")
        thresh = BRICKS.Threshold(
            old_threshold, name="Test", output_coding="temporal-L"
        )

        scaffold.add_brick(vector_1, "input")
        scaffold.add_brick(dot_brick, input_nodes=(0, 0))
        scaffold.add_brick(thresh, input_nodes=(1, 0), output=True)

        scaffold.lay_bricks()
        if self.debug:
            scaffold.summary(verbose=2)
        return scaffold

    def check_spike_output(self, spikes, expected, scaffold):
        graph_names = list(scaffold.graph.nodes.data("name"))
        before_spikes, after_spikes = spikes
        before_expected, after_expected = expected

        processed = set()
        if self.debug:
            print("Before")
        for row in before_spikes.itertuples():
            neuron_name = graph_names[int(row.neuron_number)][0]
            if self.debug:
                print(neuron_name, row.time)
            processed.add((neuron_name, row.time))

        if self.debug:
            print("After")
        for row in after_spikes.itertuples():
            neuron_name = graph_names[int(row.neuron_number)][0]
            if self.debug:
                print(neuron_name, row.time)
            processed.add((neuron_name, row.time))

        test_brick_tag = scaffold.name_to_tag["Test"]
        test_brick = scaffold.circuit.nodes[scaffold.brick_to_number[test_brick_tag]][
            "brick"
        ]
        for entry in before_expected:
            converted = (test_brick.generate_neuron_name(entry[0]), entry[1])
            if self.debug:
                print(entry, converted)
            self.assertTrue(converted in processed)
            processed.remove(converted)
        for entry in after_expected:
            converted = (test_brick.generate_neuron_name(entry[0]), entry[1])
            if self.debug:
                print(entry, converted)
            self.assertTrue(converted in processed)
            processed.remove(converted)

        assert len(processed) == 0

    # tests
    @pytest.mark.xfail()  # TODO fugu/backends/snn_backend.py:152: KeyError: 'Test'
    def test_change_neuron_properties(self):
        props = {}
        props["Test"] = {}
        props["Test"]["threshold"] = 1.0
        self.run_property_test([[0, 1, 0, 1], 3.0], [props], [[[], [("Main", 1.0)]]])

    def teardown_method(self):
        super().teardown_method()
