def get_max_magnitude_synapse_values(graph, synapse_property_names=['weight'], default_values={'weight': 1.0}):
    if len(synapse_property_names) == 0:
        return {}

    max_values = {key: 0 for key in synapse_property_names}
    for _, _, edge_data in graph.edges.data():
        for key in synapse_property_names:
            value = abs(edge_data.get(key, default_values[key]))
            if value > max_values[key]:
                max_values[key] = value

    return max_values


def get_max_magnitude_neuron_values(graph, neuron_property_names=['threshold'], default_values={'threshold': 1.0}):
    if len(neuron_property_names) == 0:
        return {}

    max_values = {key: 0 for key in neuron_property_names}
    for _, node_data in graph.nodes.data():
        for key in neuron_property_names:
            value = abs(node_data.get(key, default_values[key]))
            if value > max_values[key]:
                max_values[key] = value

    return max_values