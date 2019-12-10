#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
These bricks are used to test features/capabilities of fugu backends.
"""
from .bricks import Brick, input_coding_types


class InstantDecay(Brick):
    """
    A brick used to test neurons that have instant decay.
    """

    def __init__(self, num_inputs, name=None):
        '''
        Construtor for this brick.
        Arguments:
            + name - Name of the brick.  If not specified, a default will be used.  Name should be unique.
        '''
        super(Brick, self).__init__()
        self.is_built = False
        self.metadata = {'D': 1}
        self.name = name
        self.num_inputs = num_inputs

    def build(self, graph, metadata, control_nodes, input_lists, input_codings):
        """
        Build Dot brick.

        Arguments:
            + graph - networkx graph to define connections of the computational graph
            + metadata - dictionary to define the shapes and parameters of the brick
            + control_nodes - list of dictionary of auxillary nodes.
                Expected keys:
                    'complete' - A list of neurons that fire when the brick is done
            + input_lists - list of nodes that will contain input
            + input_coding - list of input coding formats.  ('Raster', 'Undefined' supported)

        Returns:
            + graph of a computational elements and connections
            + dictionary of output parameters (shape, coding, layers, depth, etc)
            + list dictionary of control nodes ('complete')
            + list of output edges
            + list of coding formats of output ('current')
        """

        num_inputs = sum([len(in_list) for in_list in input_lists])
        if num_inputs != self.num_inputs:
            raise ValueError(
                    "Input length does not match expected number. Expected: {}, Found: {}".format(
                                                                                             num_inputs,
                                                                                             self.num_inputs,
                                                                                             )
                    )

        graph.add_node(
                self.name + "_begin",
                threshold=0.5,
                potential=1.0,
                decay=0.0,
                index=-1,
                p=1.0,
                )
        graph.add_node(
                self.name + "_complete",
                threshold=0.5,
                potential=0.0,
                decay=0.0,
                index=-1,
                p=1.0,
                )

        graph.add_node(
                self.name + "_main",
                threshold=self.num_inputs - 0.01,
                potential=0.0,
                decay=1.0,
                index=0,
                )
        graph.add_edge(self.name + "_main", self.name + "_complete", weight=1.0, delay=1.0)

        for input_list in input_lists:
            for input_neuron in input_list:
                graph.add_edge(
                        input_neuron,
                        self.name + "_main",
                        weight=1.0,
                        delay=1.0,
                        )

        self.is_built = True
        return (graph, metadata, [{'complete': self.name + "_complete"}], [[self.name + "_main"]], input_codings)
