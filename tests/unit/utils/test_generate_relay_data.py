import networkx as nx
import numpy as np
import pytest

from fugu.utils.optimization import generate_relay_data


class TestGenerateRelayData:
    @staticmethod
    def create_test_graph(edges, delays):
        graph = nx.Graph()
        for (u, v), delay in zip(edges, delays):
            graph.add_edge(u, v, delay=delay)

        return graph

    @pytest.mark.parametrize("num_edges", range(1, 10))
    def test_no_added_relay(self, num_edges):
        delays = [1 for _ in range(num_edges)]
        edges = [(i, i + 1) for i in range(num_edges)]
        graph = TestGenerateRelayData.create_test_graph(edges, delays)
        relay_data = generate_relay_data(graph, 10)

        assert relay_data.get_total_num_relays() == 0
        for edge in edges:
            assert relay_data.get_final_delay(edge) is None

    @pytest.mark.parametrize("max_delay", np.arange(2, 63))
    def test_single_relay(self, max_delay):
        num_edges = 10
        delays = [1 for _ in range(num_edges)]
        edges = [(i, i + 1) for i in range(num_edges)]
        delays[-1] = max_delay + 1
        graph = TestGenerateRelayData.create_test_graph(edges, delays)
        relay_data = generate_relay_data(graph, max_delay)

        assert relay_data.get_total_num_relays() == 1
        assert relay_data.get_relay_count(edges[-1]) == 1
        assert relay_data.get_final_delay(edges[-1]) == 1

    @pytest.mark.parametrize("max_delay, delay_value_multiplier, expected_final_delay", [(2, 5, 1), (10, 5, 2), (10, 5, 9), (63, 3, 4), (63, 3, 30), (63, 3, 62)])
    def test_single_relay_chain_with_final_delay(self, max_delay, delay_value_multiplier, expected_final_delay):
        num_edges = 10
        delays = [1 for _ in range(num_edges)]
        edges = [(i, i + 1) for i in range(num_edges)]
        # delays[-1] = max_delay * delay_value_multiplier + delay_value_remainder
        delays[-1] = max_delay * delay_value_multiplier + expected_final_delay
        graph = TestGenerateRelayData.create_test_graph(edges, delays)
        relay_data = generate_relay_data(graph, max_delay)

        expected_total_relays = int(delays[-1] / max_delay)

        assert relay_data.get_total_num_relays() == expected_total_relays
        assert relay_data.get_relay_count(edges[-1]) == expected_total_relays
        assert relay_data.get_final_delay(edges[-1]) == expected_final_delay

    @pytest.mark.parametrize("max_delay, delay_value_multiplier", [(2, 5), (2, 9), (10, 5), (10, 2), (63, 3), (63, 30), (63, 62)])
    def test_single_relay_chain_no_final_delay(self, max_delay, delay_value_multiplier):
        num_edges = 10
        delays = [1 for _ in range(num_edges)]
        edges = [(i, i + 1) for i in range(num_edges)]
        delays[-1] = max_delay * delay_value_multiplier
        graph = TestGenerateRelayData.create_test_graph(edges, delays)
        relay_data = generate_relay_data(graph, max_delay)

        expected_total_relays = int(delays[-1] / max_delay)

        assert relay_data.get_total_num_relays() == expected_total_relays
        assert relay_data.get_relay_count(edges[-1]) == expected_total_relays
        assert relay_data.get_final_delay(edges[-1]) == 0

    def test_multiple_relay_chains(self):
        num_edges = 10
        max_delay = 63
        delays = [max_delay * (i + 1) + i + 1 for i in range(num_edges)]
        edges = [(i, i + 1) for i in range(num_edges)]

        graph = TestGenerateRelayData.create_test_graph(edges, delays)
        relay_data = generate_relay_data(graph, max_delay)
        expected_total_relays = []
        expected_final_delays = []
        for delay in delays:
            total_relays, final_delay = divmod(delay, max_delay)
            if final_delay == 0:
                total_relays -= 1
                final_delay = None
            expected_total_relays.append(total_relays)
            expected_final_delays.append(final_delay)

        assert relay_data.get_total_num_relays() == sum(expected_total_relays)
        for edge, total_relays, final_delay in zip(edges, expected_total_relays, expected_final_delays):
            assert relay_data.get_relay_count(edge) == total_relays
            assert relay_data.get_final_delay(edge) == final_delay
