"""
Sub-bricks:
    - A sub-brick is a reoccuring circuit pattern that can be modularized
        - e.g. registers that store the binary encoding of some value
    - sub-bricks comprise of two functions:
        - one that generates the nodes that make up the sub-brick
            - prefixed by "create_"
            - returns list of node names
        - one that generates the edges between a sub-brick and other elements of the circuit
            - prefixed by "connect_"

These are only really useful for people writing bricks.
"""


def create_register(graph, name, thresholds=None, decays=None, potentials=None, register_size=5, tag=None):
    thresholds_is_single_value = False
    if thresholds is None:
        thresholds = 1.0
        thresholds_is_single_value = True
    elif isinstance(thresholds, float) or isinstance(thresholds, int):
        thresholds_is_single_value = True

    decays_is_single_value = False
    if decays is None:
        decays = 0.0
        decays_is_single_value = True
    elif isinstance(decays, float) or isinstance(decays, int):
        decays_is_single_value = True

    potentials_is_single_value = False
    if potentials is None:
        potentials = 0.0
        potentials_is_single_value = True
    elif isinstance(potentials, float) or isinstance(potentials, int):
        potentials_is_single_value = True

    nodes = []
    for i in range(register_size):
        slot_name = "{}_{}".format(name, i)
        nodes.append(slot_name)
        graph.add_node(
                slot_name,
                threshold=thresholds if thresholds_is_single_value else thresholds[i],
                decay=decays if decays_is_single_value else decays[i],
                potential=potentials if potentials_is_single_value else potentials[i],
                is_register=True,
                register_index=i,
                register_tag=tag,
                register_name=name,
                )

    return nodes


def connect_register_to_register(graph, register1_name, register2_name, weights=None, delays=None, register_size=5):
    weights_is_single_value = False
    if weights is None:
        weights = 1.0
        weights_is_single_value = True
    elif isinstance(weights, float) or isinstance(weights, int):
        weights_is_single_value = True

    delays_is_single_value = False
    if delays is None:
        delays = 1.0
        delays_is_single_value = True
    elif isinstance(delays, float) or isinstance(delays, int):
        delays_is_single_value = True

    for i in range(register_size):
        i_suffix = "_{}".format(i)
        graph.add_edge(
                register1_name + i_suffix,
                register2_name + i_suffix,
                weight=weights if weights_is_single_value else weights[i],
                delay=delays if delays_is_single_value else delays[i],
                )


def connect_neuron_to_register(graph, neuron_name, register_name, weights=None, delays=None, register_size=5):
    weights_is_single_value = False
    if weights is None:
        weights = 1.0
        weights_is_single_value = True
    elif isinstance(weights, float) or isinstance(weights, int):
        weights_is_single_value = True

    delays_is_single_value = False
    if delays is None:
        delays = 1.0
        delays_is_single_value = True
    elif isinstance(delays, float) or isinstance(delays, int):
        delays_is_single_value = True

    for i in range(register_size):
        i_suffix = "_{}".format(i)
        graph.add_edge(
                neuron_name,
                register_name + i_suffix,
                weight=weights if weights_is_single_value else weights[i],
                delay=delays if delays_is_single_value else delays[i],
                )


def connect_register_to_neuron(graph, register_name, neuron_name, weights=None, delays=None, register_size=5):
    weights_is_single_value = False
    if weights is None:
        weights = 1.0
        weights_is_single_value = True
    elif isinstance(weights, float) or isinstance(weights, int):
        weights_is_single_value = True

    delays_is_single_value = False
    if delays is None:
        delays = 1.0
        delays_is_single_value = True
    elif isinstance(delays, float) or isinstance(delays, int):
        delays_is_single_value = True

    for i in range(register_size):
        i_suffix = "_{}".format(i)
        graph.add_edge(
                register_name + i_suffix,
                neuron_name,
                weight=weights if weights_is_single_value else weights[i],
                delay=delays if delays_is_single_value else delays[i],
                )
