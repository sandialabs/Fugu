#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math
import networkx as nx

from .bricks import Brick, CompoundBrick, input_coding_types
from .register_bricks import Register, Max


class SimpleGraphTraversal(Brick):
    """
    * This brick traverses a graph (using breadth first search) given a starting vertex.
    * This brick can also be used to solve single source shortest path using edge delays.
    * Predecessor/parent information is returned through edge references.
    Args:
        target_graph: NetworkX.Digraph object representing the graph to be searched
        target_node: Node in the graph that is the target of the paths
        name: Name of the brick.
            * If not specified, a default will be used. Name should be unique.
        output_coding: Output coding type, default is 'temporal-L'
    """
    def __init__(
        self,
        target_graph,
        target_node=None,
        name="SimpleGraphTraversal",
        store_parent_info=False,
        output_coding='temporal-L',
    ):

        super(SimpleGraphTraversal, self).__init__(name)
        self.is_built = False
        self.name = name
        self.supported_codings = input_coding_types
        self.target_node = target_node
        self.target_graph = target_graph
        self.output_codings = [output_coding]
        self.metadata = {'D': None}

        self.store_parent_info = store_parent_info

    def build(self, graph, metadata, control_nodes, input_lists,
              input_codings):
        """
        Build Shortest Path brick.

        Args:
            graph: networkx graph to define connections of the computational graph
                * If the graph has edge weights, this brick will solve the single source shortest paths problem
            metadata (dict): dictionary to define the shapes and parameters of the brick
            control_nodes (dict): dictionary of lists of auxillary networkx nodes.
                Expected keys:
                    'complete' - A list of neurons that fire when the brick is done
            input_lists (list): list of nodes that will contain input
            input_coding (list): list of input coding formats.  All coding types supported

        Returns:
            graph: graph of a computational elements and connections
            self.metadata: dictionary of output parameters (shape, coding, layers, depth, etc)
            complete: dictionary of control nodes ('complete')
            output_lists (list): list of output
            self.output_codings (list): list of coding formats of output
        """

        if len(input_lists) != 1:
            raise ValueError('Incorrect Number of Inputs.')
        for input_coding in input_codings:
            if input_coding not in self.supported_codings:
                raise ValueError(
                    "Unsupported Input Coding. Found: {}. Allowed: {}".format(
                        input_coding,
                        self.supported_codings,
                    ))

        begin_node_name = self.generate_neuron_name('begin')
        graph.add_node(begin_node_name,
                       threshold=0.5,
                       decay=0.0,
                       potential=0.0)
        graph.add_edge(control_nodes[0]['complete'],
                       begin_node_name,
                       weight=1.0,
                       delay=2)

        complete_name = self.generate_neuron_name('complete')
        graph.add_node(
            complete_name,
            index=len(self.target_graph.nodes),
            threshold=0.9
            if self.target_node else 1.0 * len(self.target_graph.nodes) - 0.1,
            decay=0.0,
            potential=0.0,
        )
        complete_node_list = [complete_name]

        output_node_list = []

        id_base = self.generate_neuron_name("NodeID_{}")
        parent_base = self.generate_neuron_name("ParentID_{}")
        for node in self.target_graph.nodes:
            node_name = self.generate_neuron_name(str(node))
            graph.add_node(node_name,
                           index=(node, ),
                           threshold=1.0,
                           decay=0.0,
                           potential=0.0,
                           is_vertex=True)
            graph.add_edge(node_name, node_name, weight=-10000, delay=1)
            if self.target_node:
                if node == self.target_node:
                    output_node_list.append(node_name)
                    graph.add_edge(node_name,
                                   complete_name,
                                   weight=1.0,
                                   delay=2.0)
            else:
                output_node_list.append(node_name)
                graph.add_edge(node_name, complete_name, weight=1.0, delay=2.0)

        edge_reference_names = []
        for node in self.target_graph.nodes:
            node_name = self.generate_neuron_name(str(node))
            for neighbor in self.target_graph.neighbors(node):
                # Need to scale up the delays for timing issues
                if 'weight' in self.target_graph.edges[node, neighbor]:
                    delay = 2 * self.target_graph.edges[node,
                                                        neighbor]['weight']
                else:
                    delay = 2

                neighbor_name = self.generate_neuron_name(str(neighbor))
                if self.store_parent_info:
                    reference_name = self.generate_neuron_name(
                        "{}-{}-{}".format(self.name, node, neighbor))
                    edge_reference_names.append(reference_name)

                    graph.add_node(
                        reference_name,
                        threshold=1.0,
                        decay=0.0,
                        potential=0.0,
                        from_vertex=node,
                        to_vertex=neighbor,
                        is_edge_reference=True,
                    )
                    graph.add_edge(neighbor_name,
                                   reference_name,
                                   weight=-10000,
                                   delay=1.0)
                    if node == self.target_node:
                        weight = -10000
                    else:
                        weight = 1.1
                    graph.add_edge(node_name,
                                   reference_name,
                                   weight=weight,
                                   delay=delay - 1.0)
                    graph.add_edge(reference_name,
                                   neighbor_name,
                                   weight=weight,
                                   delay=1.0)
                else:
                    if node == self.target_node:
                        graph.add_edge(node_name,
                                       neighbor_name,
                                       weight=-10000,
                                       delay=delay)
                    else:
                        graph.add_edge(node_name,
                                       neighbor_name,
                                       weight=1.1,
                                       delay=delay)

        for input_list in input_lists:
            for input_neuron in input_list:
                index = graph.nodes[input_neuron]['index']
                if type(index) is tuple:
                    index = index[0]
                if type(index) is not int:
                    raise TypeError("Neuron index should be Tuple or Int.")
                graph.add_edge(input_neuron,
                               self.generate_neuron_name(str(index)),
                               weight=2.0,
                               delay=1)

        self.is_built = True

        output_lists = [
            complete_node_list, output_node_list, edge_reference_names
        ]

        return (
            graph,
            self.metadata,
            [{
                'complete': complete_node_list,
                'begin': begin_node_name
            }],
            output_lists,
            self.output_codings,
        )


class RegisterGraphTraversal(CompoundBrick):
    """
    This brick traverses a graph (using breadth first search) given a starting vertex.
    This brick can also be used to solve single source shortest path using edge delays.
    Predecessor/parent information is returned through edge references.

    """
    def __init__(
        self,
        target_graph,
        target_node=None,
        name="RegisterGraphTraversal",
        store_parent_info=False,
        output_coding='temporal-L',
    ):
        """
        Construtor for this brick.
        Arguments:
            + target_graph - NetworkX.Digraph object representing the graph to be searched
            + target_node - Node in the graph that is the target of the paths
            + name - Name of the brick.
                If not specified, a default will be used. Name should be unique.
            + output_coding - Output coding type, default is 'temporal-L'
        """
        super(RegisterGraphTraversal, self).__init__(name)
        self.is_built = False
        self.name = name
        self.supported_codings = input_coding_types
        self.target_node = target_node
        self.target_graph = nx.DiGraph()
        for u, v, data in target_graph.edges(data=True):
            self.target_graph.add_edge(u, v, **data)
        self.output_codings = [output_coding]
        self.metadata = {'D': None}

        self.store_parent_info = store_parent_info
        self.register_size = int(math.log(len(target_graph.nodes()), 2)) + 1
        self.node_indices = {}  # map node name to node index

    def build(self, graph, metadata, control_nodes, input_lists,
              input_codings):
        """
        Build Shortest Path brick.

        Arguments:
            + graph - networkx graph to define connections of the computational graph
                If the graph has edge weights, this brick will solve the single source shortest paths problem
            + metadata - dictionary to define the shapes and parameters of the brick
            + control_nodes - dictionary of lists of auxillary networkx nodes.
                Expected keys:
                    'complete' - A list of neurons that fire when the brick is done
            + input_lists - list of nodes that will contain input
            + input_coding - list of input coding formats.  All coding types supported

        Returns:
            + graph of a computational elements and connections
            + dictionary of output parameters (shape, coding, layers, depth, etc)
            + dictionary of control nodes ('complete')
            + list of output
            + list of coding formats of output
        """

        if len(input_lists) != 1:
            raise ValueError('Incorrect Number of Inputs.')
        for input_coding in input_codings:
            if input_coding not in self.supported_codings:
                raise ValueError(
                    "Unsupported Input Coding. Found: {}. Allowed: {}".format(
                        input_coding,
                        self.supported_codings,
                    ))

        begin_node_name = self.generate_neuron_name('begin')
        graph.add_node(begin_node_name,
                       threshold=0.5,
                       decay=0.0,
                       potential=0.0)
        graph.add_edge(control_nodes[0]['complete'],
                       begin_node_name,
                       weight=1.0,
                       delay=2)

        complete_name = self.generate_neuron_name('complete')
        graph.add_node(
            complete_name,
            index=len(self.target_graph.nodes),
            threshold=0.9
            if self.target_node else 1.0 * len(self.target_graph.nodes) - 0.1,
            decay=0.0,
            potential=0.0,
        )
        complete_node_list = [complete_name]

        output_node_list = []

        id_base = "NodeID_{}"
        parent_base = "ParentID_{}"
        node_index = 0
        node_id_registers = {
        }  # map node to its id register output nodes (so we can feed them into max circuits)
        node_parent_registers = {}
        for node in self.target_graph.nodes:
            node_index += 1
            self.node_indices[node] = node_index

            node_name = self.generate_neuron_name(str(node_index))
            graph.add_node(node_name,
                           index=(node_index, ),
                           threshold=1.0,
                           decay=0.0,
                           potential=0.0,
                           is_vertex=True,
                           node=node)
            graph.add_edge(node_name, node_name, weight=-10000, delay=1)
            if self.target_node:
                if node == self.target_node:
                    output_node_list.append(node_name)
                    graph.add_edge(node_name,
                                   complete_name,
                                   weight=1.0,
                                   delay=2.0)
            else:
                output_node_list.append(node_name)
                graph.add_edge(node_name, complete_name, weight=1.0, delay=2.0)

            if self.store_parent_info:
                graph, _, _, id_register_output, _ = self.build_child(
                    Register(self.register_size,
                             initial_value=node_index,
                             name=id_base.format(node),
                             register_label=node),
                    graph,
                    {},  #meta data
                    {},  #control
                    [[], [node_name], [], []
                     ],  # input: input value, recall, clear, set
                    input_codings,
                )
                node_id_registers[node] = id_register_output[-1]

        max_circuit_time = 0
        recall_time = 0
        set_time = 0
        clear_time = 0
        for node in self.target_graph.nodes:
            node_name = self.generate_neuron_name(str(self.node_indices[node]))
            if self.store_parent_info:
                # Create max_circuit
                max_inputs = []
                for in_neighbor, _ in self.target_graph.in_edges(node):
                    max_inputs.append(node_id_registers[in_neighbor])
                # Feed output of node id registers for neighbors into max circuit

                graph, metadata, max_controls, max_circuit_output, _ = self.build_child(
                    Max(default_size=self.register_size,
                        name="Max_{}".format(node)),
                    graph,
                    {},
                    {},
                    max_inputs,
                    input_codings,
                )
                max_circuit_time = metadata['max_time']

                # Create parent id register
                # Feed output of max circuit to set parent id register
                graph, metadata, _, parent_id_register, _ = self.build_child(
                    Register(self.register_size,
                             name=parent_base.format(node),
                             register_label=node,
                             single_set=True),
                    graph,
                    {},  #meta data
                    {},  #control
                    [
                        max_circuit_output[0], [node_name], [],
                        [max_controls[0]['complete']]
                    ],  # input: input value, recall, clear, set
                    input_codings,
                )
                node_parent_registers[node] = parent_id_register[-1]
                recall_time = metadata['recall_time']
                set_time = metadata['set_time']
                clear_time = metadata['clear_time']

                self.metadata[
                    'timescale_factor'] = max_circuit_time + recall_time + set_time + clear_time
            else:
                self.metadata['timescale_factor'] = 2

            # Handle outgoing edges
            for neighbor in self.target_graph.neighbors(node):
                # Need to scale up the delays for timing issues, need to calculate how long max circuit takes
                #   i.e. need to figure out how long a major timestep takes
                if 'weight' in self.target_graph.edges[node, neighbor]:
                    if max_circuit_time > 2:
                        delay = self.target_graph.edges[node, neighbor][
                            'weight'] * (self.metadata['timescale_factor'])
                    else:
                        delay = 2 * self.target_graph.edges[node,
                                                            neighbor]['weight']
                else:
                    delay = self.metadata['timescale_factor'] if self.metadata[
                        'timescale_factor'] > 2 else 2

                neighbor_name = self.generate_neuron_name(
                    str(self.node_indices[neighbor]))
                if node == self.target_node:
                    graph.add_edge(node_name,
                                   neighbor_name,
                                   weight=-10000,
                                   delay=delay)
                else:
                    graph.add_edge(node_name,
                                   neighbor_name,
                                   weight=1.1,
                                   delay=delay)

        for input_list in input_lists:
            for input_neuron in input_list:
                index = graph.nodes[input_neuron]['index']
                if type(index) is tuple:
                    index = index[0]
                if type(index) is not int:
                    raise TypeError("Neuron index should be Tuple or Int.")
                graph.add_edge(input_neuron,
                               self.generate_neuron_name(
                                   str(self.node_indices[index])),
                               weight=2.0,
                               delay=1)

        self.is_built = True

        output_lists = [complete_node_list, output_node_list]
        if self.store_parent_info:
            for node in self.target_graph.nodes:
                output_lists.append(node_parent_registers[node])

        return (
            graph,
            self.metadata,
            [{
                'complete': complete_node_list,
                'begin': begin_node_name
            }],
            output_lists,
            self.output_codings,
        )


class FlowAugmentingPath(Brick):
    """
    This brick computes flow augmenting path based on (Ali, Kwisthout 2019)
    """
    def __init__(
        self,
        flow_graph,
        name="FlowAugmentingPath",
        output_coding='temporal-L',
    ):
        """
        Construtor for this brick.
        Arguments:
            + flow_graph - NetworkX.Digraph object representing the flow graph
            + name - Name of the brick.
                If not specified, a default will be used. Name should be unique.
            + output_coding - Output coding type, default is 'temporal-L'
        """
        super(FlowAugmentingPath, self).__init__(name)
        self.is_built = False
        self.name = name
        self.supported_codings = input_coding_types
        self.output_codings = [output_coding]
        self.capacity_base = "C_({},{})"
        self.residual_base = "H_({},{})"
        self.recall_base = "R_({},{})"

        self.flow_graph = flow_graph

        self.metadata = {'D': None}

    def set_properties(self, graph, properties):
        """
        Returns an updated version of the graph based on the parameter values passed.
        """
        num_edges = len(self.flow_graph.edges())
        initial_potential = num_edges + 1
        flow_values = {}
        for edge in properties['flow']:
            key = self.generate_neuron_name(self.capacity_base.format(*edge))
            flow_values[key] = properties['flow'][edge] + initial_potential
        nx.set_node_attributes(graph, name='potential', values=flow_values)
        return graph

    def build(self, graph, metadata, control_nodes, input_lists,
              input_codings):
        """
        Build Flow Augmenting Path brick.

        Arguments:
            + graph - networkx graph to define connections of the computational graph
                If the graph has edge weights, this brick will solve the single source shortest paths problem
            + metadata - dictionary to define the shapes and parameters of the brick
            + control_nodes - dictionary of lists of auxillary networkx nodes.
                Expected keys:
                    'complete' - A list of neurons that fire when the brick is done
            + input_lists - list of nodes that will contain input
            + input_coding - list of input coding formats.  All coding types supported

        Returns:
            + graph of a computational elements and connections
            + dictionary of output parameters (shape, coding, layers, depth, etc)
            + dictionary of control nodes ('complete')
            + list of output
            + list of coding formats of output
        """

        if len(input_lists) != 1:
            raise ValueError('Incorrect Number of Inputs.')
        for input_coding in input_codings:
            if input_coding not in self.supported_codings:
                raise ValueError(
                    "Unsupported Input Coding. Found: {}. Allowed: {}".format(
                        input_coding,
                        self.supported_codings,
                    ))

        num_edges = len(self.flow_graph.edges())
        initial_potential = num_edges + 1

        begin_node_name = self.generate_neuron_name('begin')
        graph.add_node(begin_node_name,
                       threshold=0.5,
                       decay=0.0,
                       potential=0.0)
        graph.add_edge(control_nodes[0]['complete'],
                       begin_node_name,
                       weight=1.0,
                       delay=2)

        complete_name = self.generate_neuron_name('complete')
        graph.add_node(
            complete_name,
            index=len(self.flow_graph.edges),
            threshold=0.9,
            decay=0.0,
            potential=0.0,
        )
        complete_node_list = [complete_name]

        output_node_list = []

        capacity_neurons = []
        curr_index = 1
        for u, v, props in self.flow_graph.edges(data=True):
            capacity_name = self.generate_neuron_name(
                self.capacity_base.format(u, v))
            residual_name = self.generate_neuron_name(
                self.residual_base.format(u, v))
            recall_name = self.generate_neuron_name(
                self.recall_base.format(u, v))
            graph.add_node(
                capacity_name,
                index=curr_index,
                threshold=props['capacity'] + initial_potential - 0.01,
                decay=0.0,
                potential=props['flow'] + initial_potential,
                edge=(u, v),
                neuron_type='capacity',
            )
            curr_index += 1
            graph.add_node(
                residual_name,
                index=curr_index,
                threshold=1 + initial_potential - 0.01,
                decay=0.0,
                potential=initial_potential,
                edge=(u, v),
                neuron_type='residual',
            )
            curr_index += 1
            graph.add_node(
                recall_name,
                index=curr_index,
                threshold=1 + initial_potential - 0.01,
                decay=0.0,
                potential=initial_potential,
                edge=(u, v),
                neuron_type='recall',
            )
            curr_index += 1
            graph.add_edge(
                capacity_name,
                residual_name,
                weight=-1 * initial_potential,
                delay=1.0,
            )
            graph.add_edge(
                capacity_name,
                recall_name,
                weight=-1 * initial_potential,
                delay=1.0,
            )

        for node in self.flow_graph.nodes():
            in_neighbors = self.flow_graph.pred[node]
            out_neighbors = self.flow_graph.succ[node]
            for in_neighbor in in_neighbors:
                for out_neighbor in out_neighbors:
                    residual_ab = self.generate_neuron_name(
                        self.residual_base.format(in_neighbor, node))
                    residual_cd = self.generate_neuron_name(
                        self.residual_base.format(node, out_neighbor))
                    graph.add_edge(
                        residual_cd,
                        residual_ab,
                        weight=1.0,
                        delay=1.0,
                    )

                    recall_ab = self.generate_neuron_name(
                        self.recall_base.format(in_neighbor, node))
                    recall_cd = self.generate_neuron_name(
                        self.recall_base.format(node, out_neighbor))
                    graph.add_edge(
                        recall_ab,
                        recall_cd,
                        weight=1.0,
                        delay=1.0,
                    )
            if 's' in in_neighbors:
                graph.add_edge(
                    self.generate_neuron_name(
                        self.residual_base.format('s', node)),
                    self.generate_neuron_name(
                        self.recall_base.format('s', node)),
                    weight=1.0,
                    delay=1.0,
                )
            if 't' in out_neighbors:
                graph.add_edge(
                    begin_node_name,
                    self.generate_neuron_name(
                        self.residual_base.format(node, 't')),
                    weight=1.0,
                    delay=1.0,
                )

        for input_list in input_lists:
            for input_neuron in input_list:
                index = graph.nodes[input_neuron]['index']
                if type(index) is tuple:
                    if len(index) > 0:
                        index = index[0]
                    else:
                        continue
                elif type(index) is not int:
                    raise TypeError("Neuron index should be Tuple or Int.")
                graph.add_edge(input_neuron,
                               begin_node_name,
                               weight=2.0,
                               delay=1)

        self.is_built = True

        output_lists = [complete_node_list, output_node_list]

        return (
            graph,
            self.metadata,
            [{
                'complete': complete_node_list,
                'begin': begin_node_name
            }],
            output_lists,
            self.output_codings,
        )
