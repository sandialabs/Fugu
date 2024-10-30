import networkx as nx
import numpy as np
import pytest

from fugu.utils.optimization import DelayRelayData


class TestDelayRelayData:
    def setup_method(self):
        self.different_edge = (1, 2)

    def test_initialize_default(self):
        delay_relay_data = DelayRelayData()
        assert delay_relay_data.first_relay == 0
        assert delay_relay_data.relay_count == 0
        assert delay_relay_data.final_delay == 0

    def test_initialize_one_field(self):
        delay_relay_data = DelayRelayData(first_relay=1)
        assert delay_relay_data.first_relay == 1
        assert delay_relay_data.relay_count == 0
        assert delay_relay_data.final_delay == 0

    def test_initialize_two_field(self):
        delay_relay_data = DelayRelayData(first_relay=1, relay_count=2)
        assert delay_relay_data.first_relay == 1
        assert delay_relay_data.relay_count == 2
        assert delay_relay_data.final_delay == 0

    def test_initialize_all_fields(self):
        delay_relay_data = DelayRelayData(first_relay=1, relay_count=2, final_delay=19)
        assert delay_relay_data.first_relay == 1
        assert delay_relay_data.relay_count == 2
        assert delay_relay_data.final_delay == 19
