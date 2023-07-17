import numpy as np
from brick_test import BrickTest

import fugu.bricks as BRICKS
from fugu.backends import snn_Backend
from fugu.scaffold import Scaffold


class TestSnnRegister(BrickTest):
    def setup_method(self):
        super().setup_method()
        self.backend = snn_Backend()

    # Base class overrides
    def build_scaffold(self, input_values):
        register_size, initial_value, action = input_values

        scaffold = Scaffold()

        converted_input = self.convert_input(action)

        value_brick = BRICKS.Vector_Input(
            converted_input["value"], coding="Raster", name="Value"
        )
        scaffold.add_brick(value_brick, "input")
        recall_brick = BRICKS.Vector_Input(
            converted_input["recall"],
            coding="Raster",
            name="Recall",
            time_dimension=True,
        )
        scaffold.add_brick(recall_brick, "input")
        clear_brick = BRICKS.Vector_Input(
            converted_input["clear"], coding="Raster", name="Clear", time_dimension=True
        )
        scaffold.add_brick(clear_brick, "input")
        set_brick = BRICKS.Vector_Input(
            converted_input["set"], coding="Raster", name="Set", time_dimension=True
        )
        scaffold.add_brick(set_brick, "input")

        register_brick = BRICKS.Register(
            register_size, initial_value=initial_value, name="Register"
        )

        scaffold.add_brick(
            register_brick, input_nodes=[(0, 0), (1, 0), (2, 0), (3, 0)], output=True
        )

        scaffold.lay_bricks()

        if self.debug:
            scaffold.summary(verbose=2)

        return scaffold

    def calculate_max_timesteps(self, input_values):
        return 12

    def check_spike_output(self, spikes, expected, scaffold):
        value = 0
        graph_names = list(scaffold.graph.nodes.data("name"))
        for row in spikes.sort_values("time").itertuples():
            neuron_name = graph_names[int(row.neuron_number)][0]
            if self.debug:
                print(neuron_name, row.time)
            if "Register" in neuron_name and "output" in neuron_name:
                index = int(neuron_name.split("_")[-1])
                value += 2**index

        assert expected == value

    def convert_input(self, input_values):
        inputs = {}
        if ":" in input_values:
            action, new_value = input_values.split(":")
            new_value = [int(b) for b in "{:05b}".format(int(new_value))[::-1]]
            inputs["value"] = np.array(new_value)
        else:
            action = input_values
            inputs["value"] = np.array([0])
        timing = 0
        if action == "recall":
            inputs["recall"] = np.array([[1]])
            inputs["clear"] = np.array([[0]])
            inputs["set"] = np.array([[0]])
        elif action == "clear":
            inputs["recall"] = np.array([[0, 0, 0, 1]])
            inputs["clear"] = np.array([[1, 0, 0, 0]])
            inputs["set"] = np.array([[0, 0, 0, 0]])
        elif action == "set":
            inputs["recall"] = np.array([[0, 0, 0, 0, 0, 1]])
            inputs["clear"] = np.array([[0, 0, 0, 0, 0, 0]])
            inputs["set"] = np.array([[1, 0, 0, 0, 0, 0]])

        return inputs

    def test_register_recall(self):
        self.basic_test([5, 22, "recall"], 22)

    def test_register_clear(self):
        self.basic_test([5, 17, "clear"], 0)

    def test_register_set(self):
        self.basic_test([5, 3, "set:19"], 19)

    def teardown_method(self):
        super().teardown_method()
