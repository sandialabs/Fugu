#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import networkx as nx
import numpy as np
from warnings import warn
from .port import ChannelSpec, PortSpec, ChannelData, PortData, PortUtil

class Scaffold:
    """
    Class to handle a scaffold of bricks
    """

    supported_backends = ['snn', 'ds_legacy', 'snn_legacy', 'pynn']

    def __init__(self):
        self.circuit = nx.DiGraph()
        self.pos = {}
        self.count = {}
        self.graph = None
        self.is_built = False
        self.metrics = None
        self.brick_to_number = {}
        self.tag_to_name = {}
        self.name_to_tag = {}

    def add_brick(self,
                  brick,
                  input_nodes=[],
                  metadata=None,
                  name=None,
                  output=False):
        """
        Add a brick to the scaffold.
        Newer code that calls this function should only pass the brick itself.
        The job of connecting ports between bricks is handled separately by calls to connect().
        Older code that calls this function may pass values for the other arguments.
        These will be automatically translated into equivalent port connections.

        Args:
            brick (obj): object of type brick
            input_nodes (list): list of node numbers (Default: [])
            metadata :  dictionary of shapes and parameters of the brick (Default: None)
            name (any): string of the brick's name (Default: none)
            output (bool): bool flag to indicate if a brick is an output brick (Default: False)

        Returns:
            The brick. This allows you to collapse brick creation and addition scaffold into single
            call, while retaining a reference to the created object for making connections. Example:
                A = scaffold.add_brick(SomeBrick())

        Raises:
            ValueError: Raises if node name is already used.
        """

        # Ensure that brick has a unique name within the circuit graph.
        if name is None:
            if brick.name is None:
                brick_type = str(type(brick))
                if brick_type not in self.count: self.count[brick_type] = 0
                name = f"{brick_type}_{self.count[brick_type]}"
                self.count[brick_type] = self.count[brick_type] + 1
            else:
                name = brick.name
        elif name in self.circuit.nodes:
            raise ValueError("Node name already used.")
        if name in self.name_to_tag:
            raise ValueError("Node name already used.")

        if brick.name is None:
            brick.name = name

        tag = brick.brick_tag
        self.tag_to_name[tag] = name
        self.name_to_tag[name] = tag

        to_key = self.circuit.number_of_nodes()
        self.circuit.add_node(
            to_key,
            tag=tag,
            name=name,
            brick=brick,
        )  # ,metadata=metadata)
        self.brick_to_number[tag] = to_key
        to_node = self.circuit.nodes[to_key]
        to_brick = to_node['brick']
        to_ports = to_brick.input_ports()

        if not to_ports and to_brick.output_ports():  # This brick has output ports but no input ports, meaning it is a source in the graph.
            to_node['layer'] = 'input'
        if output: to_node['layer'] = 'output'

        # The remaining code in this function deals with legacy bricks ...

        # Make sure we're working with a list of inputs
        if type(input_nodes) is not list:
            input_nodes = [input_nodes]

        # The legacy input_nodes contains a list of references to previously-added bricks in the circuit graph.
        # Each brick gets an integer key when it is added. The simplest reference is just this integer, indicating
        # "whatever brick n outputs". In cases where a brick can output more than one thing, the reference is a
        # tuple, where the first element is the brick key and the second is an output position.
        # Numbering of bricks starts from zero. This allows us to use a couple of magic numbers for convenience:
        # -1 refers to the immediate predecessor brick.
        # -2 refers to nothing, effectively identifying this brick as an originator of input.
        for to_index, from_key in enumerate(input_nodes):
            from_index = 0
            if from_key == 'input':   # The string 'input' is equivalent to -2, meaning nothing (no input).
                from_key = -2
            elif from_key == -1:  # Translate -1 into the penultimate brick key, whatever that is.
                from_key = to_key - 1
            elif type(from_key) is str:  # It's also possible to reference a bick by its unique name (assigned above).
                from_key = self.brick_to_number[from_key]
            elif type(from_key) is tuple:
                from_index = from_key[1]
                if type(from_key[0]) is str:  # The brick reference inside the tuple can be a string ...
                    from_key = self.brick_to_number[from_key[0]]
                else:                         # or an int. If it's anything else, we're in trouble.
                    from_key = from_key[0]

            if from_key < -1:
                to_node['layer'] = 'input'
            else:
                # Convert channel numbers to port names, if possible.
                from_node = self.circuit.nodes[from_key]
                from_brick = from_node['brick']
                from_ports = from_brick.output_ports()
                from_port = PortUtil.find_port_name(from_ports, from_index)
                to_port   = PortUtil.find_port_name(to_ports,   to_index)
                self.connect(from_brick, to_brick, from_port, to_port)

        return brick

    def connect(self, from_brick, to_brick, from_port='0', to_port='0'):
        """
        Binds an output port to an input port in the circuit graph.
        The port names are determined by their respective bricks.
        These can be queried by calling input_ports() and output_ports() on a given brick class.

        For binding auto-ports, you may either use an explicit suffix or specify the base name
        without a suffix. For example, if name="input", then you can bind to "input3" explicitly,
        or bind to "input". In the latter case, the exact suffix will be determined during lay_bricks().
        If you want to establish several different connections between the same pair of bricks,
        it may be necessary to specify the target port suffix explicitly.
        """
        if from_port == '0':
            ports = from_brick.output_ports()
            if ports:
                from_port = PortUtil.find_port_name(ports, 0)
                if not from_port: raise ValueError('Brick lacks a default output port.')
        if to_port == '0':
            ports = to_brick.input_ports()
            if ports:
                to_port = PortUtil.find_port_name(ports, 0)
                if not to_port: raise ValueError('Brick lacks a default input port.')
        from_key = self.brick_to_number[from_brick.brick_tag]
        to_key   = self.brick_to_number[to_brick  .brick_tag]
        self.circuit.add_edge(from_key, to_key)
        e = self.circuit.edges[from_key, to_key]
        if not 'bind' in e: e['bind'] = {}
        bind = e['bind']
        if to_port in bind: raise ValueError('Attempt to bind more than one output port to same input port.')
        bind[to_port] = from_port

    def _assign_brick_tags(self, built_graph, tag, field='brick'):
        new_nodes = [
            new_node for new_node, node_value in built_graph.nodes(data=True)
            if field not in node_value
        ]
        for new_node in new_nodes:
            built_graph.nodes[new_node]['brick'] = tag
        return built_graph

    def lay_bricks(self, verbose=0):
        """
        Build a computational graph that can be used by the backend.

        Args:
            verbose (int): value to specify level of verbosity (Default: 0 to indicate None)

        Returns:
            built_graph: networkX diGraph
        """
        self.graph = nx.DiGraph()
        G = self.graph
        C = self.circuit

        # Handle Input Nodes
        if verbose > 0: print("Laying Input Bricks.")
        for to_key, to_node in C.nodes.data():
            if to_node.get('layer') != 'input': continue;
            to_node['ports'] = to_node['brick'].build2(G)
            self._assign_brick_tags(G, to_node['tag'])
            if verbose > 0: print("Completed: ", to_key)

        # Handle all other nodes
        use_temporary = False
        limit = len(C.nodes) + 2  # Limit number of iterations to size of circuit.
        iteration = 0
        while True:
            iteration += 1
            if iteration > limit: raise RuntimeError(f'Exceeded {limit} iterations. The circuit graph has unresolvable dependencies.')

            all_nodes_built = True  # until proven false
            some_node_built = False
            for to_key, to_node in C.nodes.data():
                to_brick = to_node['brick']
                if to_brick.is_built: continue

                # Collect inputs
                inputs = {}
                all_inputs_available = True  # until proven false
                to_port_specs = to_brick.input_ports()
                for from_key, _, e in C.in_edges(nbunch=to_key, data=True):
                    from_node = C.nodes[from_key]
                    from_outputs = from_node.get('ports')
                    if not from_outputs:
                        all_inputs_available = False
                        if not use_temporary: break
                        continue
                    for to_port_name, from_port_name in e['bind'].items():
                        existing = inputs.get(to_port_name)
                        if existing:  # More than one source is bound to the same destination, which is an error.
                            # This is a slightly different condition than one tested for in connect(),
                            # because it checks across all inputs, not just inputs from one brick.
                            raise ValueError('Attempt to bind more than one output port to same input port.')

                        # Check for auto-port naming
                        to_port_spec = to_port_specs.get(to_port_name)
                        if to_port_spec and to_port_spec.maximum != 1:  # modify to_port_name to occupy a free slot
                            i = 1
                            base = to_port_name
                            to_port_name = base + str(i)
                            while inputs.get(to_port_name):
                                i = i + 1
                                to_port_name = base + str(i)

                        from_port = from_outputs.get(from_port_name)
                        if not from_port: raise ValueError('Requested output port does not exist.')
                        inputs[to_port_name] = from_port

                # Check if brick is buildable.
                if not all_inputs_available:
                    if use_temporary and not to_node.get('ports'):  # Temporary outputs have not yet been contructed.
                        port_specs = to_brick.output_shape(inputs)
                        if PortUtil.all_shapes_known(port_specs):
                            # Create temporary populations
                            ports = PortUtil.make_ports_from_specs(port_specs)
                            to_node['ports'] = ports
                            index = 0
                            for port in ports.values():
                                for channel in port.channels.values():
                                    count = 1
                                    for size in channel.spec.shape: count *= size
                                    for i in range(count):
                                        name = to_brick.generate_neuron_name(f'"{index}')  # '"' is an attempt to use a character that no sane brick developer would ever duplicate. TODO: use globally-unique integers as neuron names in all brick build functions.
                                        index += 1
                                        G.add_node(name, index=i)
                                        channel.neurons.append(name)
                    all_nodes_built = False
                    continue
                some_node_built = True

                # Check for limits on number of bindings.
                for to_port_spec in to_port_specs.values():
                    # Count number of bindings
                    count = 0
                    for key in inputs:
                        if PortUtil.autoport_match(to_port_spec.name, key): count += 1
                    # Check bounds
                    if to_port_spec.minimum and count < to_port_spec.minimum:
                        raise ValueError(f"Not enough bindings for input {to_port_spec.name} on brick {to_brick.name}. Required {to_port_spec.minimum}, but found {count}.")
                    if to_port_spec.maximum and count > to_port_spec.maximum:
                        raise ValueError(f"Too many bindings for input {to_port_spec.name} on brick {to_brick.name}. Limit {to_port_spec.maximum}, but found {count}.")

                # lay brick
                if verbose > 0: print('Laying Brick: ', to_key)
                temp_ports = to_node.get('ports')
                actual_ports = to_brick.build2(G, inputs)
                to_node['ports'] = actual_ports
                self._assign_brick_tags(G, to_node['tag'])
                if temp_ports:  # Replace temporary neurons with actual ones.
                    # We assume a 1-to-1 mapping between each temporary population and the real one generated by build2().
                    for port_name, actual_port in actual_ports.items():
                        temp_port = temp_ports[port_name]
                        for channel_name, actual_channel in actual_port.channels.items():
                            temp_channel = temp_port.channels[channel_name]
                            for actual, temp in zip(actual_channel.neurons, temp_channel.neurons):
                                # Now we have two graph keys. Need to copy the edges from temp to actual.
                                for _, n2, e in G.edges(nbunch=temp, data=True):
                                    G.add_edge(actual, n2, **e)
                                G.remove_node(temp)
                if verbose > 0: print("Complete.")

            if all_nodes_built: break
            if not some_node_built: use_temporary = True

        # Give every neuron in graph a unique ID
        # TODO: We should just use a global ID for everything.
        for i, n in enumerate(G.nodes):
            G.nodes[n]['neuron_number'] = i
        self.is_built = True
        return G

    def summary(self, verbose=0):
        """
        Display a summary of the scaffold.
        Prints information about the scaffold.
        Args:
             verbose (int): verbosity level can be 0, 1 or >1 (Default: 0)
        """

        C = self.circuit
        G = self.graph

        print("Scaffold is built: {}".format(self.is_built))
        print("-------------------------------------------------------")
        print("Bricks:")
        print()
        for i, name in enumerate(C.nodes):
            node = C.nodes[name]
            print(f"Brick No.: {i}")
            print(f"Brick Tag: {node['tag']}")
            print(f"Brick Name: {self.tag_to_name[node['tag']]}")
            print(node)
            print(f"Brick is built: {node['brick'].is_built}")
            print()
        print("-------------------------------------------------------")
        print("Brick Edges:")
        print()
        for b1, b2, e in self.circuit.edges(data=True):
            print(f"({b1}, {b2}) {e}")

        if verbose > 0 and self.graph:
            print()
            print("-------------------------------------------------------")
            print("Neurons:")
            print()
            print("Neuron Number | Neuron Name | Neuron Properties")
            for name, neuron in self.graph.nodes.data():
                print(f"{neuron['neuron_number']} | {name} | {neuron}")
            print()
            print("-------------------------------------------------------")
            print("Synapses:")
            print()
            if verbose > 1: print("Synapse Between | Synapse Properties")
            for n1, n2, e in G.edges(data=True):
                print(f"({n1}, {n2})", end="")
                if verbose > 1: print(f" {e}")
                else:           print()

        print("-------------------------------------------------------")
