import fugu.bricks as BRICKS
from fugu.backends import snn_Backend
from fugu.scaffold import Scaffold

from ..brick_test import BrickTest


class TestSnnVectorInput(BrickTest):
    def setup_method(self):
        super().setup_method()
        self.backend = snn_Backend()
        self.backend_args = {"record": "all"}
        self.brick_name = "VectorInput"

    # Base class function
    def build_scaffold(self, input_values):
        input_spikes, use_time = input_values
        scaffold = Scaffold()

        vector_brick = BRICKS.Vector_Input(
            input_spikes,
            coding="Raster",
            name=self.brick_name,
            time_dimension=use_time,
        )

        scaffold.add_brick(vector_brick, input_nodes=["input"])

        scaffold.lay_bricks()
        return scaffold

    def check_spike_output(self, spikes, expected, scaffold):
        graph_names = list(scaffold.graph.nodes.data("name"))
        fire_list = [[] for n in expected]
        for row in spikes.itertuples():
            neuron_name = graph_names[int(row.neuron_number)][0]
            neuron_index = scaffold.graph.nodes[neuron_name]["index"]
            if type(neuron_index) is not int:
                index = neuron_index[0]
                fire_list[index].append(int(row.time))

        assert expected == fire_list

    # tests
    def test_spike_array(self):
        input_spikes = [1, 1, 0, 1]
        self.basic_test((input_spikes, False), [[0], [0], [], [0]])

    def test_time_spike_array(self):
        input_spikes = [[1, 1, 0], [0, 0, 1]]
        self.basic_test((input_spikes, True), [[0, 1], [2]])

    def teardown_method(self):
        super().teardown_method()
