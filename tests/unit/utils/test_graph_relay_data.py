import networkx as nx
import numpy as np
import pytest

from fugu.utils.optimization import DelayRelayData, GraphRelayData


class TestGraphDelayRelayData:
    def setup_method(self):
        self.graph_relay_data = GraphRelayData()
        self.edge = (0, 1)

    def test_initialize_graph_relay_data(self):
        assert self.graph_relay_data.get_total_num_relays() == 0
        assert not self.graph_relay_data.has_relay_data(self.edge)
        assert self.graph_relay_data.get_relay_data(self.edge) is None

    def test_get_total_num_relays_single_edge(self):
        relay_data = DelayRelayData(relay_count=5)
        self.graph_relay_data.add_relay_data(self.edge, relay_data)

        assert self.graph_relay_data.has_relay_data(self.edge)
        assert self.graph_relay_data.get_total_num_relays() == 5

    def test_get_final_delay(self):
        relay_data = DelayRelayData(final_delay=10)
        self.graph_relay_data.add_relay_data(self.edge, relay_data)

        assert self.graph_relay_data.has_relay_data(self.edge)
        assert self.graph_relay_data.get_final_delay(self.edge) == 10

    def test_get_first_relay(self):
        relay_data = DelayRelayData(first_relay=10)
        self.graph_relay_data.add_relay_data(self.edge, relay_data)

        assert self.graph_relay_data.has_relay_data(self.edge)
        assert self.graph_relay_data.get_first_relay(self.edge) == 10

    def test_get_relay_count(self):
        relay_data = DelayRelayData(relay_count=10)
        self.graph_relay_data.add_relay_data(self.edge, relay_data)

        assert self.graph_relay_data.has_relay_data(self.edge)
        assert self.graph_relay_data.get_relay_count(self.edge) == 10

    def test_add_same_edge(self):
        relay_data1 = DelayRelayData(first_relay=10)
        relay_data2 = DelayRelayData(final_delay=9)

        self.graph_relay_data.add_relay_data(self.edge, relay_data1)
        self.graph_relay_data.add_relay_data(self.edge, relay_data2)

        assert self.graph_relay_data.has_relay_data(self.edge)
        assert self.graph_relay_data.get_first_relay(self.edge) == 0
        assert self.graph_relay_data.get_final_delay(self.edge) == 9
