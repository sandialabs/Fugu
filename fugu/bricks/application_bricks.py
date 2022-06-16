#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from .bricks import Brick, input_coding_types


class LIS(Brick):
    """
    This brick calculates the length of the longest common subsequence for a given sequence of numbers
    Construtor for this brick.
        Args:
            sequence_length: sequence_length - size of the sequence
            name: Name of the brick.  If not specified, a default will be used.  Name should be unique.
            output_coding: Output coding type, default is 'temporal-L'
    """
    def __init__(self,
                 sequence_length,
                 name="LIS",
                 output_coding='temporal-L'):

        super(LIS, self).__init__(name)
        self.is_built = False
        self.name = name
        self.supported_codings = input_coding_types

        self.output_codings = [output_coding]
        self.metadata = {'D': None}

        if sequence_length < 2:
            raise ValueError("Cannot have a sequence of only 1 element")
        self.sequence_length = sequence_length

    def build(self, graph, metadata, control_nodes, input_lists,
              input_codings):
        """
        Build LIS brick.

        Args:
            graph: networkx graph to define connections of the computational graph
            metadata (dict): dictionary to define the shapes and properties of the brick
            control_nodes (dict): dictionary of lists of auxillary networkx nodes.
                Expected keys:
                    'complete' - A list of neurons that fire when the brick is done
            input_lists (list): list of nodes that will contain input
            input_coding (list): list of input coding formats.  All coding types supported

        Returns:
            graph: graph of a computational elements and connections
            self.metadata.complete: dictionary of output properties (shape, coding, layers, depth, etc)
            self.metadata.begin: dictionary of control nodes ('complete')
            output_list (list): list of output
            self.output_codings: list of coding formats of output
        """

        if len(input_lists) != 1:
            raise ValueError('Incorrect Number of Inputs.')
        for input_coding in input_codings:
            if input_coding not in self.supported_codings:
                raise ValueError(
                    "Unsupported Input Coding. Found: {}. Allowed: {}".format(
                        input_coding, self.supported_codings))

        new_begin_node_name = self.generate_neuron_name('begin')
        graph.add_node(
            new_begin_node_name,
            threshold=0.1,
            decay=0.0,
            potential=0.0,
        )
        graph.add_edge(
            control_nodes[0]['complete'],
            new_begin_node_name,
            weight=1.0,
            delay=1.0,
        )

        complete_name = self.generate_neuron_name('complete')
        graph.add_node(
            complete_name,
            index=self.sequence_length,
            threshold=0.1,
            decay=0.0,
            potential=0.0,
        )
        complete_node_list = [complete_name]

        min_runtime = self.sequence_length

        output_Ls = []
        levels = [[] for i in range(self.sequence_length)]
        for i in range(self.sequence_length):
            L_name = self.generate_neuron_name("L_{}_Main".format(i + 1))
            graph.add_node(L_name, threshold=0.90, decay=0.0, potential=0.0)
            graph.add_edge(L_name, L_name, weight=-10000.0, delay=1.0)
            output_Ls.append(L_name)

        for i in range(self.sequence_length):
            column_a = []
            column_b = []
            x_name = self.generate_neuron_name("x_{}".format(i))
            L0_A_name = self.generate_neuron_name("L_1-x_{}-A".format(i))

            # create x_i neuron
            graph.add_node(x_name, threshold=0.0, decay=0.0, potential=0.0)

            # create column
            graph.add_node(L0_A_name, threshold=0.90, decay=0.0, potential=0.0)

            graph.add_edge(x_name, L0_A_name, weight=1.0, delay=1.0)
            graph.add_edge(L0_A_name, L0_A_name, weight=-18, delay=1.0)

            graph.add_edge(L0_A_name,
                           self.generate_neuron_name("L_1_Main"),
                           weight=1.0,
                           delay=1.0)

            levels[0].append(L0_A_name)

            for j in range(i):
                L_B_name = self.generate_neuron_name("L_{}-x_{}-B".format(
                    j + 1, i))
                L_A_name = self.generate_neuron_name("L_{}-x_{}-A".format(
                    j + 2, i))
                graph.add_node(L_B_name,
                               threshold=0.90,
                               decay=0.0,
                               potential=0.0)
                graph.add_node(L_A_name,
                               threshold=1.90,
                               decay=0.0,
                               potential=0.0)

                # Alarms
                graph.add_edge(x_name, L_B_name, weight=-19.0, delay=(j + 2.0))
                graph.add_edge(x_name, L_A_name, weight=1.0, delay=1.0)

                graph.add_edge(L_B_name, L_A_name, weight=1.0, delay=1.0)
                graph.add_edge(L_A_name, L_A_name, weight=-18.0, delay=1.0)

                graph.add_edge(L_A_name,
                               self.generate_neuron_name(
                                   "L_{}_Main".format(j + 2)),
                               weight=1.0,
                               delay=1.0)

                levels[j].append(L_B_name)
                levels[j + 1].append(L_A_name)

        for level in levels:
            if len(level) > 1:
                graph.add_edge(level[0], level[1], weight=1.0, delay=1.0)
                graph.add_edge(level[0], level[2], weight=1.0, delay=1.0)
                for i in range(1, len(level) - 2, 2):
                    graph.add_edge(level[i],
                                   level[i + 2],
                                   weight=1.0,
                                   delay=1.0)
                    graph.add_edge(level[i],
                                   level[i + 3],
                                   weight=1.0,
                                   delay=1.0)

        x_index = 0
        for input_list in input_lists:
            for input_neuron in input_list:
                graph.add_edge(input_neuron,
                               self.generate_neuron_name(
                                   "x_{}".format(x_index)),
                               weight=1.0,
                               delay=1.0)
                x_index += 1
                if x_index > self.sequence_length:
                    raise TypeError("Too many inputs to brick: {}".format(
                        self.name))

        self.is_built = True

        # Remember, bricks can have more than one output, so we need a list of list of output neurons
        output_lists = [complete_node_list, output_Ls]

        return (
            graph,
            self.metadata,
            [{
                'complete': complete_node_list[0],
                'begin': new_begin_node_name
            }],
            output_lists,
            self.output_codings,
        )
