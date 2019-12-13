#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math
import networkx as nx

from .bricks import DynamicBrick, Brick, input_coding_types
from .sub_bricks import create_register, connect_register_to_register, connect_neuron_to_register


class Graph_Traversal(Brick):
    """
    This brick traverses a graph (using breadth first search) given a starting vertex.
    This brick can also be used to solve single source shortest path using edge delays.

    """
    def __init__(
          self,
          target_graph,
          target_node=None,
          name=None,
          store_edge_references=False,
          store_parent_info=False,
          output_coding='temporal-L',
          ignore_edge_weights=False,
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
        super(Brick, self).__init__()
        self.is_built = False
        self.name = name
        self.supported_codings = input_coding_types
        self.target_node = target_node
        self.target_graph = target_graph
        self.output_codings = [output_coding]
        self.metadata = {'D': None}

        self.store_edge_references = store_edge_references
        self.store_parent_info = store_parent_info

        # whether or not to ignore edge weights of the original graph
        self.ignore_edge_weights = ignore_edge_weights

        self.register_size = int(math.ceil(math.log(len(self.target_graph.nodes()), 2)))

    def build(self, graph, metadata, control_nodes, input_lists, input_codings):
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
                                                                             )
                        )

        begin_node_name = self.name + '_begin'
        graph.add_node(begin_node_name, threshold=0.5, decay=0.0, potential=0.0)
        graph.add_edge(control_nodes[0]['complete'], begin_node_name, weight=1.0, delay=2)

        complete_name = self.name + '_complete'
        graph.add_node(
                complete_name,
                index=len(self.target_graph.nodes),
                threshold=0.9 if self.target_node else 1.0 * len(self.target_graph.nodes) - 0.1,
                decay=0.0,
                potential=0.0,
                )
        complete_node_list = [complete_name]

        output_node_list = []

        id_base = "ID_{}_{}"
        parent_base = "ParentID_{}_{}"
        for node in self.target_graph.nodes:
            node_name = self.name + str(node)
            graph.add_node(node_name, index=(node,), threshold=1.0, decay=0.0, potential=0.0, is_vertex=True)
            graph.add_edge(node_name, node_name, weight=-1000, delay=1)
            if self.target_node:
                if node == self.target_node:
                    output_node_list.append(node_name)
                    graph.add_edge(node_name, complete_name, weight=1.0, delay=2.0)
            else:
                output_node_list.append(node_name)
                graph.add_edge(node_name, complete_name, weight=1.0, delay=2.0)

            if self.store_parent_info:
                # Create registers
                binary_id = "".join(["{:0", str(self.register_size), "b}"]).format(node)
                id_potentials = [1.0 if bit == '1' else 0.0 for bit in binary_id[::-1]]
                create_register(
                  graph,
                  "ID_{}".format(node_name),
                  thresholds=1.99,
                  potentials=id_potentials,
                  register_size=self.register_size,
                  tag=node_name,
                  )
                parent_register = create_register(
                                    graph,
                                    "ParentID_{}".format(node_name),
                                    thresholds=0.99,
                                    register_size=self.register_size,
                                    tag=node_name,
                                    )

                for node in parent_register:
                    output_node_list.append(node)

                # connect neuron to id register
                connect_neuron_to_register(
                  graph,
                  node_name,
                  "ID_{}".format(node_name),
                  register_size=self.register_size,
                  )

        edge_reference_names = []
        reference_index = len(self.target_graph.nodes) + 1
        for node in self.target_graph.nodes:
            node_name = self.name + str(node)
            for neighbor in self.target_graph.neighbors(node):
                # Need to scale up the delays for timing issues
                if not self.ignore_edge_weights and 'weight' in self.target_graph.edges[node, neighbor]:
                    delay = 2 * self.target_graph.edges[node, neighbor]['weight']
                else:
                    delay = 2
                neighbor_name = self.name + str(neighbor)
                if self.store_edge_references:
                    reference_name = "{}-{}-{}".format(self.name, node, neighbor)
                    edge_reference_names.append(reference_name)

                    graph.add_node(
                            reference_name,
                            index=reference_index,
                            threshold=1.0,
                            decay=0.0,
                            potential=0.0,
                            from_vertex=node,
                            to_vertex=neighbor,
                            is_edge_reference=True,
                            )
                    graph.add_edge(neighbor_name, reference_name, weight=-1000, delay=1.0)
                    if node == self.target_node:
                        weight = -1000
                    else:
                        weight = 1.1
                    graph.add_edge(node_name, reference_name, weight=weight, delay=delay - 1.0)
                    graph.add_edge(reference_name, neighbor_name, weight=weight, delay=1.0)
                    reference_index += 1
                else:
                    if node == self.target_node:
                        graph.add_edge(node_name, neighbor_name, weight=-1000, delay=delay)
                    else:
                        graph.add_edge(node_name, neighbor_name, weight=1.1, delay=delay)
                if self.store_parent_info:
                    connect_register_to_register(
                      graph,
                      "ID_{}".format(node_name),
                      "ParentID_{}".format(neighbor_name),
                      delays=delay - 1,
                      register_size=self.register_size,
                      )

        for input_list in input_lists:
            for input_neuron in input_list:
                index = graph.nodes[input_neuron]['index']
                if type(index) is tuple:
                    index = index[0]
                if type(index) is not int:
                    raise TypeError("Neuron index should be Tuple or Int.")
                graph.add_edge(input_neuron, self.name + str(index), weight=2.0, delay=1)

        self.is_built = True

        output_lists = [complete_node_list, output_node_list, edge_reference_names]

        return (
                 graph,
                 self.metadata,
                 [{'complete': complete_node_list, 'begin': begin_node_name}],
                 output_lists,
                 self.output_codings,
                 )


class Flow_Augmenting_Path(DynamicBrick):
    """
    This brick computes flow augmenting path based on (Ali, Kwisthout 2019)
    """
    def __init__(
          self,
          flow_graph,
          name=None,
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
        super(Brick, self).__init__()
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
            flow_values[self.capacity_base.format(*edge)] = properties['flow'][edge] + initial_potential
        nx.set_node_attributes(graph, name='potential', values=flow_values)
        return graph

    def build(self, graph, metadata, control_nodes, input_lists, input_codings):
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
                                                                             )
                        )

        num_edges = len(self.flow_graph.edges())
        initial_potential = num_edges + 1

        begin_node_name = self.name + '_begin'
        graph.add_node(begin_node_name, threshold=0.5, decay=0.0, potential=0.0)
        graph.add_edge(control_nodes[0]['complete'], begin_node_name, weight=1.0, delay=2)

        complete_name = self.name + '_complete'
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
            capacity_name = self.capacity_base.format(u, v)
            residual_name = self.residual_base.format(u, v)
            recall_name = self.recall_base.format(u, v)
            graph.add_node(
                    capacity_name,
                    index=curr_index,
                    threshold=props['capacity'] + initial_potential - 0.01,
                    decay=0.0,
                    potential=props['flow'] + initial_potential,
                    edge=(u,v),
                    neuron_type='capacity',
                    )
            curr_index += 1
            graph.add_node(
                    residual_name,
                    index=curr_index,
                    threshold=1 + initial_potential - 0.01,
                    decay=0.0,
                    potential=initial_potential,
                    edge=(u,v),
                    neuron_type='residual',
                    )
            curr_index += 1
            graph.add_node(
                    recall_name,
                    index=curr_index,
                    threshold=1 + initial_potential - 0.01,
                    decay=0.0,
                    potential=initial_potential,
                    edge=(u,v),
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
                    residual_ab = self.residual_base.format(in_neighbor, node)
                    residual_cd = self.residual_base.format(node, out_neighbor)
                    graph.add_edge(
                            residual_cd,
                            residual_ab,
                            weight=1.0,
                            delay=1.0,
                            )

                    recall_ab = self.recall_base.format(in_neighbor, node)
                    recall_cd = self.recall_base.format(node, out_neighbor)
                    graph.add_edge(
                            recall_ab,
                            recall_cd,
                            weight=1.0,
                            delay=1.0,
                            )
            if 's' in in_neighbors:
                graph.add_edge(
                        self.residual_base.format('s', node),
                        self.recall_base.format('s', node),
                        weight=1.0,
                        delay=1.0,
                        )
            if 't' in out_neighbors:
                graph.add_edge(
                        begin_node_name,
                        self.residual_base.format(node, 't'),
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
                graph.add_edge(input_neuron, begin_node_name, weight=2.0, delay=1)

        self.is_built = True

        output_lists = [complete_node_list, output_node_list]

        return (
                 graph,
                 self.metadata,
                 [{'complete': complete_node_list, 'begin': begin_node_name}],
                 output_lists,
                 self.output_codings,
                 )
