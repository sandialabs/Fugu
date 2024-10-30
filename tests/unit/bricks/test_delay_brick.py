import numpy as np
import pytest

from fugu.backends.snn_backend import snn_Backend
from fugu.bricks import Delay, Vector_Input
from fugu.scaffold import Scaffold


class Test_Delay:
    @pytest.mark.parametrize(
        "delay",
        [1, 5, 10, 30, 63, 64, 65, 70, 88, 90, 100, 115, 120, 128, 200, 300, 1000],
    )
    def test_delay(self, delay):
        scaffold = Scaffold()
        scaffold.add_brick(
            Vector_Input(np.array([[1]])),
            "input",
        )

        scaffold.add_brick(
            Delay(delay, name="Delay"),
            [(0, 0)],
            output=True,
        )

        scaffold.lay_bricks()

        backend_args = {}

        backend = snn_Backend()
        backend.compile(scaffold, backend_args)
        result = backend.run(delay + 5)

        print("Time \t Neuron Name")
        graph_names = list(scaffold.graph.nodes.data("name"))
        start_time = 0
        end_time = 0
        for row in result.itertuples():
            neuron_name = graph_names[int(row.neuron_number)][0]
            if "src" in neuron_name:
                start_time = row[1]
            if "dest" in neuron_name:
                end_time = row[1]
            print(f"{row[1]} \t {neuron_name}")

        assert end_time - start_time == delay
