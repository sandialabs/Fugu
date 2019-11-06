#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 14:46:55 2019

@author: smusuva
"""
import math

import bricks


class Shortest_Path(bricks.Brick):
    '''
    This brick provides a single-source shortest path determination.
    Expects a single input where the index corresponds to the node number on the graph.

    '''
    def __init__(
          self,
          target_graph,
          target_node=None,
          name=None,
          store_edge_references=False,
          output_coding='temporal-L',
          ):
        '''
        Construtor for this brick.
        Arguments:
            + target_graph - NetworkX.Digraph object representing the graph to be searched
            + target_node - Node in the graph that is the target of the paths
            + name - Name of the brick.
                If not specified, a default will be used. Name should be unique.
            + output_coding - Output coding type, default is 'temporal-L'
        '''
        super(bricks.Brick, self).__init__()
        # The brick hasn't been built yet.
        self.is_built = False
        # We just store the name passed at construction.
        self.name = name
        # For this example, we'll let any input coding work even though the answer might not make sense.
        self.supported_codings = bricks.input_coding_types
        # Right now, we'll convert node labels to integers in the order of
        # graph.nodes() However, in the fugure, this should be improved to be more flexible.
        self.target_node = None
        for i, node in enumerate(target_graph.nodes()):
            if node is target_node:
                self.target_node = i
        self.target_graph = target_graph
        self.output_codings = [output_coding]
        self.metadata = {'D': None}
        self.store_edge_references = store_edge_references

        self.register_size = math.ceil(math.log(len(self.target_graph.nodes()), 2))

    def build(self, graph, metadata, control_nodes, input_lists, input_codings):
        """
        Build Shortest Path brick.

        Arguments:
            + graph - networkx graph to define connections of the computational graph
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
                raise ValueError("Unsupported Input Coding. Found: {}. Allowed: {}".format(
                                                                                      input_coding,
                                                                                      self.supported_codings,
                                                                                    ))

        # All bricks should provide a neuron that spikes when the brick has completed processing.
        # We just put in a basic relay neuron that will spike when it recieves any spike from its
        # single input, which is the complete_node from the first input.
        # All nodes we add to the graph should have basic neuron parameters (threshold, decay)
        # Reasonable defaults will be filled-in, but these defaults may depend on the execution platform.
        # Additionally, nodes should have a field called 'index' which is a local index used to reference the
        # position of the node.  This can be used by downstream bricks.  A simple example might be
        # a 3-bit binary representation will add 3 nodes to the graph with indices 0,1,2
        # We do have to do some work to establish best practices here.
        # new_complete_node_name = self.name + '_complete'
        # graph.add_node(new_complete_node_name,
        #               index = -1,
        #              threshold = 0.0,
        #              decay =0.0,
        #              p=1.0,
        #              potential=0.0)
        # complete_node = [new_complete_node_name]
        new_begin_node_name = self.name + '_begin'
        graph.add_node(
                new_begin_node_name,
                threshold=0.5,
                decay=0.0,
                potential=0.0,
                )
        graph.add_edge(
                control_nodes[0]['complete'],
                new_begin_node_name,
                weight=1.0,
                delay=2,
                )

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
        for node in self.target_graph.nodes:
            node_name = self.name + str(node)
            graph.add_node(
                    node_name,
                    index=(node,),
                    threshold=1.0,
                    decay=0.0,
                    potential=0.0,
                    is_vertex=True,
                    )
            graph.add_edge(node_name, node_name, weight=-1000, delay=1)
            if self.target_node:
                if node == self.target_node:
                    output_node_list.append(node_name)
                    graph.add_edge(node_name, complete_name, weight=1.0, delay=2.0)
            else:
                output_node_list.append(node_name)
                graph.add_edge(node_name, complete_name, weight=1.0, delay=2.0)

        edge_reference_names = []
        reference_index = len(self.target_graph.nodes) + 1
        for node in self.target_graph.nodes:
            node_name = self.name + str(node)
            for neighbor in self.target_graph.neighbors(node):
                # delay = weight + 1 works for ds
                # delay = 2 * weight - 1 works for pynn-brian
                # delay = 2 * weight works for spynn
                delay = 2 * self.target_graph.edges[node, neighbor]['weight']
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
                        graph.add_edge(node_name, reference_name, weight=-1000, delay=delay - 1)
                        graph.add_edge(reference_name, neighbor_name, weight=-1000, delay=1)
                    else:
                        graph.add_edge(node_name, reference_name, weight=1.1, delay=delay - 1.0)
                        graph.add_edge(reference_name, neighbor_name, weight=1.1, delay=1.0)
                    reference_index += 1
                else:
                    if node == self.target_node:
                        graph.add_edge(node_name, neighbor_name, weight=-1000, delay=delay)
                    else:
                        graph.add_edge(node_name, neighbor_name, weight=1.1, delay=delay)

        for input_neuron in input_lists[0]:
            index = graph.nodes[input_neuron]['index']
            if type(index) is tuple:
                index = index[0]
            if type(index) is not int:
                raise TypeError("Neuron index should be Tuple or Int.")
            graph.add_edge(
                    input_neuron,
                    self.name + str(index),
                    weight=2.0,
                    delay=1,
                    )

        self.is_built = True

        # Remember, bricks can have more than one output, so we need a list of list of output neurons
        output_lists = [complete_node_list, output_node_list, edge_reference_names]

        return (
                 graph,
                 self.metadata,
                 [{'complete': complete_node_list, 'begin': new_begin_node_name}],
                 output_lists,
                 self.output_codings,
                 )


class Breadth_First_Search(bricks.Brick):
    '''
    This brick performs a BFS traversal.
    Expects a single input where the index corresponds to the node number on the graph.

    '''
    def __init__(
          self,
          target_graph,
          target_node=None,
          store_edge_references=False,
          name=None,
          output_coding='temporal-L',
          ):
        '''
        Construtor for this brick.
        Arguments:
            + target_graph - NetworkX.Digraph object representing the graph to be searched
            + target_node - Node in the graph that is the target of the paths
            + name - Name of the brick.  If not specified, a default will be used.  Name should be unique.
            + output_coding - Output coding type, default is 'temporal-L'
        '''
        super(bricks.Brick, self).__init__()
        # The brick hasn't been built yet.
        self.is_built = False
        # We just store the name passed at construction.
        self.name = name
        # For this example, we'll let any input coding work even though the answer might not make sense.
        self.supported_codings = bricks.input_coding_types
        # Right now, we'll convert node labels to integers in the order of
        # graph.nodes() However, in the fugure, this should be improved to be
        # more flexible.
        self.target_node = target_node
        self.target_graph = target_graph
        self.output_codings = [output_coding]
        self.metadata = {'D': None}

        # mappings of the original graph to the embedded graph and vice-versa
        # used primarily to interpret what the spikes mean
        self.neuron_vertex_map = {}
        self.vertex_neuron_map = {}
        self.edge_synapse_map = {}
        self.synapse_edge_map = {}

        self.store_edge_references = store_edge_references

    def get_neuron_vertex_map(self):
        return self.neuron_vertex_map

    def get_vertex_neuron_map(self):
        return self.vertex_neuron_map

    def get_synapse_edge_map(self):
        return self.synapse_edge_map

    def get_edge_synapse_map(self):
        return self.edge_synapse_map

    def build(self, graph, metadata, control_nodes, input_lists, input_codings):
        """
        Build BFS brick.

        Arguments:
            + graph - networkx graph to define connections of the computational graph
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
                raise ValueError("Unsupported Input Coding. Found: {}. Allowed: {}".format(input_coding,
                                                                                           self.supported_codings))

        # All bricks should provide a neuron that spikes when the brick has completed processing.
        # We just put in a basic relay neuron that will spike when it recieves any spike from its
        # single input, which is the complete_node from the first input.
        # All nodes we add to the graph should have basic neuron parameters (threshold, decay)
        # Reasonable defaults will be filled-in, but these defaults may depend on the execution platform.
        # Additionally, nodes should have a field called 'index' which is a local index used to reference the
        # position of the node.  This can be used by downstream bricks.  A simple example might be
        # a 3-bit binary representation will add 3 nodes to the graph with indices 0,1,2
        # We do have to do some work to establish best practices here.

        default_delay = 2.0
        new_begin_node_name = self.name + '_begin'
        graph.add_node(
                new_begin_node_name,
                threshold=0.5,
                decay=0.0,
                potential=0.0,
                )
        graph.add_edge(
                control_nodes[0]['complete'],
                new_begin_node_name,
                weight=1.0,
                delay=default_delay,
                )

        complete_name = self.name + '_complete'
        graph.add_node(
                complete_name,
                index=len(self.target_graph.nodes),
                threshold=0.9 if self.target_node else 1.0 * len(self.target_graph.nodes)-.1,
                decay=0.0,
                potential=0.0,
                )
        complete_node_list = [complete_name]

        target_node_list = []
        for node in self.target_graph.nodes:
            node_name = self.name + str(node)

            self.neuron_vertex_map[node] = node
            self.vertex_neuron_map[node] = node

            graph.add_node(
                    node_name,
                    index=(node,),
                    threshold=0.9,
                    decay=0.0,
                    potential=0.0,
                    is_vertex=True,
                    )
            if self.store_edge_references:
                graph.add_edge(node_name, node_name, weight=-1000, delay=1.0)
            else:
                graph.add_edge(node_name, node_name, weight=-1000, delay=default_delay)

            if self.target_node:
                if node == self.target_node:
                    target_node_list.append(node_name)
                    graph.add_edge(node_name, complete_name, weight=1.0, delay=default_delay)
            else:
                target_node_list.append(node_name)
                graph.add_edge(node_name, complete_name, weight=1.0, delay=default_delay)

        edge_reference_names = []
        reference_index = len(self.target_graph.nodes) + 1
        for node in self.target_graph.nodes:
            node_name = self.name + str(node)
            neighbors = list(self.target_graph.neighbors(node))
            for neighbor in neighbors:
                neighbor_name = self.name + str(neighbor)
                if self.store_edge_references:
                    reference_name = "{}-{}-{}".format(self.name, node, neighbor)

                    self.synapse_edge_map[reference_index] = (node, neighbor)
                    self.edge_synapse_map[(node, neighbor)] = reference_index

                    edge_reference_names.append(reference_name)

                    graph.add_node(
                            reference_name,
                            index=reference_index,
                            threshold=0.9,
                            decay=0.0,
                            potential=0.0,
                            from_vertex=node,
                            to_vertex=neighbor,
                            is_edge_reference=True,
                            )

                    if self.target_node and node == self.target_node:
                        weight = -1000
                        delay = 1.0
                    else:
                        weight = 1.0
                        delay = default_delay

                    graph.add_edge(node_name, reference_name, weight=weight, delay=delay)
                    graph.add_edge(reference_name, neighbor_name, weight=weight, delay=delay)
                    graph.add_edge(neighbor_name, reference_name, weight=-1000, delay=1.0)
                    reference_index += 1
                else:
                    graph.add_edge(node_name, neighbor_name, weight=1.0, delay=default_delay)

        for input_neuron in input_lists[0]:
            index = graph.nodes[input_neuron]['index']
            if type(index) is tuple:
                index = index[0]
            if type(index) is not int:
                raise TypeError("Neuron index should be Tuple or Int.")

            if self.name + str(index) in graph.nodes():
                graph.add_edge(
                        input_neuron,
                        self.name + str(index),
                        weight=1.0,
                        delay=default_delay,
                        )

        self.is_built = True

        # Remember, bricks can have more than one output, so we need a list of list of output neurons
        output_lists = [complete_node_list, edge_reference_names, target_node_list]

        return (
                 graph,
                 self.metadata,
                 [{'complete': complete_node_list[0], 'begin': new_begin_node_name}],
                 output_lists,
                 self.output_codings,
                 )
