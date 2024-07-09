import networkx as nx
import numpy as np
import pytest

from fugu.utils.optimization import DelayRelayData


class TestDelayRelayData:
    def setup_method(self):
        self.delay_relay_data = DelayRelayData()
        self.edge = (0, 1)

    def test_initialize_delay_relay_data(self):
        assert self.delay_relay_data.get_total_relays() == 0
        assert self.delay_relay_data.get_final_delay(self.edge) is None

    @pytest.mark.parametrize("num_relays", [1, 2, 5, 10, 100])
    def test_add_multiple_relays(self, num_relays):
        for i in range(num_relays):
            self.delay_relay_data.add_relay(self.edge, i)

        assert self.delay_relay_data.get_total_relays() == num_relays
        assert self.delay_relay_data.get_relay_list(self.edge) == list(range(num_relays))

    def test_no_final_delay(self):
        self.delay_relay_data.add_relay(self.edge, 0)
        assert self.delay_relay_data.get_final_delay(self.edge) is None

    @pytest.mark.parametrize("final_delay", [0, 1, 5, 10, 64, 128])
    def test_set_final_delay(self, final_delay):
        self.delay_relay_data.add_final_delay(self.edge, final_delay)
        assert self.delay_relay_data.get_final_delay(self.edge) == final_delay