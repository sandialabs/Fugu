from collections import deque


def CalculateSpikeTimes(circuit, main_key='timestep'):
    initial_spikes = {}
    input_nodes = [node for node in circuit.nodes if ('layer' in circuit.nodes[node])
                   and (circuit.nodes[node]['layer'] == 'input')]
    max_steps = 0
    for input_node in input_nodes:
        for timestep, spike_list in enumerate(circuit.nodes[input_node]['brick']):
            if timestep > max_steps:
                max_steps = timestep
    if main_key == 'timestep':
        for i in range(0, max_steps + 1):
            initial_spikes[i] = deque()
        for input_node in input_nodes:
            for timestep, spike_list in enumerate(circuit.nodes[input_node]['brick']):
                if len(spike_list) > 0:
                    initial_spikes[timestep].extend(spike_list)
    elif main_key == 'neuron_name':
        for input_node in input_nodes:
            for timestep, spike_list in enumerate(circuit.nodes[input_node]['brick']):
                for neuron in spike_list:
                    if neuron not in initial_spikes:
                        initial_spikes[neuron] = []
                    initial_spikes[neuron].append(timestep)
    else:
        raise ValueError("main_key argument must be 'timestep' or 'neuron_name', not {}".format(main_key))

    return initial_spikes


