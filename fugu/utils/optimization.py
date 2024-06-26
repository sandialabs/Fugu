import networkx as nx


def offset_voltages(graph, desired_reset_value=0.0):
    # Shift's neurons reset potentials to desired_reset_value then offset's a neuron's threshold, starting potential 
    for _, node in graph.nodes.data():
        current_reset_value = node.get('reset_voltage', desired_reset_value)

        voltage_offset = current_reset_value - desired_reset_value

        Vinit  = node.get('potential', 0.0)
        Vspike = node.get('threshold', 1.0)

        node['reset_voltage'] = desired_reset_value
        node['potential']     = Vinit - voltage_offset
        node['threshold']     = Vspike - voltage_offset

    return graph


class DelayRelayData:
    def __init__(self):
        self.relay_list = {}
        self.final_delays = {}

    def add_relay(self, edge, relay):
        if edge not in self.relay_list:
            self.relay_list[edge] = []
        self.relay_list[edge].append(relay)

    def add_final_delay(self, edge, final_delay):
        self.final_delays[edge] = final_delay

    def has_relay_list(self, edge):
        return edge in self.relay_list

    def get_relay_list(self, edge):
        if edge not in self.relay_list:
            return None
        return self.relay_list[edge]

    def get_all_relay_lists(self):
        return self.relay_list

    def get_final_delay(self, edge):
        if edge not in self.final_delays:
            return None
        return self.final_delays[edge]

    def get_total_relays(self):
        return sum([len(self.relay_list[e]) for e in self.relay_list])


def generate_relay_data(graph : nx.Graph, max_delay) -> DelayRelayData:
    # Returns data that can be used to create relay neurons. 
    # This assumes delay is handled by the destination rather than the source
    delay_relay_data = DelayRelayData()
    total_num_relays = 0
    for (u, v) in graph.edges():
        delay = graph[u][v]["delay"]
        if delay > max_delay:
            next = total_num_relays
            while delay > max_delay:
                delay_relay_data.add_relay((u, v), next)
                total_num_relays += 1
                next += 1
                delay -= max_delay
            delay_relay_data.add_final_delay((u, v), delay)

    return delay_relay_data