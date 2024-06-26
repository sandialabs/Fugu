import numpy as np
import pytest

from fugu.backends.lava_backend import Loihi2HWInterface, lava_Backend
from fugu.bricks import Delay, Vector_Input
from fugu.scaffold import Scaffold


class Test_UtilFunctions:
    def test_calculateBitLength(self):
        pass


class Test_HW_Interface:
    def test_hw_threshold_bit_limit(self):
        interface = Loihi2HWInterface(duration=10)

        assert interface.threshold_bit_limit == 16


class Test_lava_Backend:
    @pytest.mark.parametrize(
        "delay",
        [1, 5, 10, 30, 63, 64, 65, 70, 88, 90, 100, 115, 120, 128, 200, 300, 1000],
    )
    @pytest.mark.parametrize("loihi_config", ["sim2", "hw2"])
    def test_delay_chain(self, delay, loihi_config):
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
        backend_args["config"] = loihi_config

        backend = lava_Backend()
        backend.compile(scaffold, backend_args)
        result = backend.run(delay + 9)

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

    @pytest.mark.parametrize(
        "delay_values",
        [
            ([5, 5, 5]),
            ([5, 65, 5]),
            ([5, 65, 65]),
            ([5, 5, 165]),
            ([5, 5, 1650]),
            ([5, 165, 165]),
            ([5, 165, 1650]),
            ([5, 1650, 1650]),
            ([65, 65, 65]),
            ([65, 65, 165]),
            ([65, 65, 1650]),
            ([65, 165, 165]),
            ([65, 165, 1650]),
            ([65, 1650, 1650]),
            ([165, 165, 165]),
            ([165, 165, 1650]),
            ([165, 1650, 1650]),
            ([1650, 1650, 1650]),
         ],
    )
    @pytest.mark.parametrize("loihi_config", ["sim2", "hw2"])
    def test_delay_values(self, delay_values, loihi_config):
        scaffold = Scaffold()
        scaffold.add_brick(
            Vector_Input(np.array([[1]])),
            "input",
        )

        for index, delay in enumerate(delay_values):
            scaffold.add_brick(
                Delay(delay, name=f"{index}>Delay"),
                [(0, 0)],
                output=True,
            )

        scaffold.lay_bricks()

        backend_args = {}
        backend_args["config"] = loihi_config

        backend = lava_Backend()
        backend.compile(scaffold, backend_args)
        result = backend.run(max(delay_values) + 9)

        print("Time \t Neuron Name")
        graph_names = list(scaffold.graph.nodes.data("name"))
        start_times = [0 for _ in delay_values]
        end_times = [0 for _ in delay_values]
        for row in result.itertuples():
            neuron_name = graph_names[int(row.neuron_number)][0]
            if "src" in neuron_name or "dest" in neuron_name:
                index = int(neuron_name.split(">")[0])
                if "src" in neuron_name:
                    start_times[index] = row[1]
                if "dest" in neuron_name:
                    end_times[index] = row[1]
            print(f"{row[1]} \t {neuron_name}")

        results = []
        for start, end, expected in zip(start_times, end_times, delay_values):
            results.append(end - start == expected)
        assert all(results)