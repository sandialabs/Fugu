from dataclasses import dataclass

import networkx as nx


def offset_voltages(graph, desired_reset_value=0.0):
    # Shift's neurons reset potentials to desired_reset_value then offset's a neuron's threshold, starting potential
    for _, node in graph.nodes.data():
        current_reset_value = node.get("reset_voltage", desired_reset_value)

        voltage_offset = current_reset_value - desired_reset_value

        Vinit  = node.get("potential", 0.0)
        Vspike = node.get("threshold", 1.0)

        node["reset_voltage"] = desired_reset_value
        node["potential"]     = Vinit - voltage_offset
        node["threshold"]     = Vspike - voltage_offset

    return graph


@dataclass(frozen=True)
class DelayRelayData:
    first_relay: int = 0
    relay_count: int = 0
    final_delay: int = 0


class GraphRelayData:
    def __init__(self):
        self._relay_data = {}

    def add_relay_data(self, edge, relay_data: DelayRelayData):
        self._relay_data[edge] = relay_data

    def get_relay_data(self, edge):
        return self._relay_data.get(edge, None)

    def has_relay_data(self, edge):
        return edge in self._relay_data

    def get_total_num_relays(self):
        return sum([drd.relay_count for drd in self._relay_data.values()])

    def get_final_delay(self, edge):
        if edge not in self._relay_data:
            return None
        return self._relay_data[edge].final_delay

    def get_first_relay(self, edge):
        if edge not in self._relay_data:
            return None
        return self._relay_data[edge].first_relay

    def get_relay_count(self, edge):
        if edge not in self._relay_data:
            return None
        return self._relay_data[edge].relay_count


def generate_relay_data(graph: nx.DiGraph, max_delay) -> GraphRelayData:
    # Returns data that can be used to create relay neurons.
    # This assumes delay is handled by the destination rather than the source
    graph_relay_data = GraphRelayData()
    current_start_relay = 0
    for u, v in graph.edges():
        delay = graph[u][v]["delay"]
        if delay > max_delay:
            start_relay = current_start_relay
            number_of_relays, final_delay = divmod(delay, max_delay)
            graph_relay_data.add_relay_data((u, v), DelayRelayData(first_relay=start_relay, relay_count=number_of_relays, final_delay=final_delay))
            current_start_relay += number_of_relays

    return graph_relay_data


def insert_relay_neurons(G: nx.DiGraph, max_delay) -> nx.DiGraph:
    # Replace edges with large delays with relay neurons
    print("Replacing edges with large delays with chains of relay neurons")
    edges_to_expand = []
    for n1, n2, edge in G.edges.data():
        delay = edge['delay']
        if delay > max_delay:
            edges_to_expand.append((n1, n2))

    for n1, n2 in edges_to_expand:
        edge_data = G.edges[(n1, n2)]
        delay = edge_data['delay']
        weight = edge_data['weight']

        relay_count = 0


        current_node = n1
        next_node = f"{n1}-{n2}-relay-node:{relay_count}"
        while delay > max_delay:
            G.add_node(
                        next_node,
                        threshold=0.5,
                        decay=1.0,
                        )
            G.add_edge(
                        current_node,
                        next_node,
                        weight=1.0,
                        delay=max_delay,
                        )
            delay -= max_delay
            current_node = next_node
            relay_count += 1
            next_node = f"{n1}-{n2}-relay-node:{relay_count}"

        G.add_edge(
                    current_node,
                    n2,
                    weight=weight,
                    delay=delay,
                    )
    G.remove_edges_from(edges_to_expand)