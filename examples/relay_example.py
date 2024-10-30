# Template functions: these are functions your backend should implement/handle
def create_neuron(n1, threshold, potential, decay):
    pass

def connect(src, dest, delay, weight):
    pass


# "Main"
# This is a rough outline of how the backend's "compile" function uses the relay data
# NOTE: This is just an example. Please take your specific platform into account
neurons = {}
for node in G.nodes:
    threshold = G.nodes[node]['threshold']
    potential = G.nodes[node]['potential']
    decay = G.nodes[node]['decay']
    neurons[node] = create_neuron(node, threshold, potential, decay)
    
relay_data = generate_relay_data(G, 63)
# create relay neurons
for relay in relay_data.get_total_num_relays():
    threshold = 0.1
    potential = 0.0
    decay = 1 # Instant decay

    neuron_name = f"Relay_{relay}"
    neurons[neuron_name] = create_neuron(neuron_name, threshold, potential, decay)

for n1, n2, edge in G.edges.data():
    delay = edge.get('delay', 1)
    weight = edge.get('weight', 1)

    if relay_data.has_relay_data((n1, n2)):
        # Do stuff with the relay
        num_relays = relay_data.get_relay_count((n1, n2))

        first_relay = relay_data.get_first_relay((n1, n2))
        first_relay_name = f"Relay_{first_relay}"

        connect(n1, first_relay_name, delay, weight)

        for relay_index in range(num_relays - 1):
            current_relay = relay_index + first_relay 
            next_relay = current_relay + 1
            current_relay_name = f"Relay_{current_relay}"
            next_relay_name = f"Relay_{next_relay}"
            connect(current_relay_name, next_relay_name, delay=1, weight=1)

        final_relay = first_relay + num_relays - 1 # need to offset by one due to indexing
        final_relay_name = f"Relay_{final_relay}"
        connect(final_relay, n2, delay, weight)
    else:
        connect(n1, n2, delay, weight)
