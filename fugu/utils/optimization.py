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