#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 14:46:55 2019

@author: smusuva
"""
import numpy as np

import bricks


class PRN(bricks.Brick):
    """
    Psuedo-random neuron brick.
    Generates spikes randomly (a uniform random [0,1] draw is compared against a threshold).
    """

    def __init__(self, probability=0.5, steps=None, shape=(1,), name=None, output_coding='Undefined'):
        '''
        Constructor for this brick.
        Arguments:
            + probability - Probability of a spike at any timestep
            + steps - Number of timesteps to produce spikes. None provides un-ending output.
            + shape - shape of the neurons in the brick
            + output_coding - Desired output coding for the brick
        '''
        super(bricks.Brick, self).__init__()
        self.is_built = False
        self.metadata = {}
        self.probability = probability
        self.name = name
        self.shape = shape
        self.steps = steps
        self.output_coding = output_coding
        self.supported_codings = bricks.input_coding_types

    def build(self, graph, metadata, control_nodes, input_lists, input_codings):
        if len(input_lists) == 0:
            raise ValueError("PRN brick requires at least 1 input.")
        # Driver Neuron
        driver_neuron = self.name+'_driver'
        graph.add_node(driver_neuron, threshold=0.7, decay=1.0)
        graph.add_edge(driver_neuron, driver_neuron, weight=1.0, delay=1)
        # PRNs
        output_list = []
        for neuron_index in np.ndindex(self.shape):
            output_neuron = self.name+'_' + str(neuron_index)
            graph.add_node(output_neuron, threshold=0.7, decay=1.0, p=self.probability)
            output_list.append(output_neuron)
            graph.add_edge(driver_neuron, output_neuron, weight=1.0, delay=1)
        complete_neuron = self.name+'_complete'
        complete_threshold = self.steps - 1.1 if self.steps is not None else 1.0
        graph.add_node(complete_neuron, threshold=complete_threshold, decay=0.0)
        if self.steps is not None:
            graph.add_edge(driver_neuron, complete_neuron, weight=1.0, delay=1)
            graph.add_edge(complete_neuron, driver_neuron, weight=-10.0, delay=1)
        for input_control_nodes in control_nodes:
            graph.add_edge(input_control_nodes['complete'], driver_neuron, weight=1.0, delay=1)
        self.is_built = True
        return (graph, self.metadata, [{'complete': complete_neuron}], [output_list], [self.output_coding])


class Threshold(bricks.Brick):
    """
    Class to handle Threshold bricks.Brick. Inherits from bricks.Brick
    """

    def __init__(self, threshold, decay=0.0, p=1.0, name=None, output_coding=None):
        '''
        Construtor for this brick.
        Arguments:
            + threshold - Threshold value.  For input coding 'current', float.  For 'temporal-L', int.
            + decay - Decay value for threshold neuron ('current' input only)
            + p - Probability of firing when exceeding threshold ('current' input only)
            + name - Name of the brick.  If not specified, a default will be used.  Name should be unique.
            + output_coding - Force a return of this output coding.  Default is 'unary-L'
        '''
        super(bricks.Brick, self).__init__()
        self.is_built = False
        self.metadata = {}
        self.name = name
        self.p = p
        self.decay = decay
        self.threshold = threshold
        self.output_coding = output_coding
        self.supported_codings = ['current', 'Undefined', 'temporal-L']

    def build(self, graph, metadata, control_nodes, input_lists, input_codings):
        """
        Build Threshold brick.

        Arguments:
            + graph - networkx graph to define connections of the computational graph
            + metadata - dictionary to define the shapes and parameters of the brick
            + control_nodes - list of dictionary of auxillary nodes.
                  Expected keys:
                      'complete' - A neurons that fire when the brick is done
                      'begin' - A neurons that first when the brick begins processing (for temporal coded inputs)
            + input_lists - list of nodes that will contain input
            + input_coding - list of input coding formats

        Returns:
            + graph of a computational elements and connections
            + dictionary of output parameters (shape, coding, layers, depth, etc)
            + list dictionary of control nodes ('complete')
            + list of output
            + list of coding formats of output
        """

        if len(input_codings) != 1:
            raise ValueError("Only one input is permitted.")
        if input_codings[0] not in self.supported_codings:
            raise ValueError(
                    "Input coding not supported. Expected: {} ,Found: {}".format(
                                                                            self.supported_codings,
                                                                            input_codings[0],
                                                                            )
                    )
        if input_codings[0] == 'current' or input_codings[0] == 'Undefined':
            graph.add_node(
                    self.name,
                    threshold=self.threshold,
                    decay=self.decay,
                    p=self.p,
                    )
            for edge in input_lists[0]:
                graph.add_edge(
                        edge['source'],
                        self.name,
                        weight=edge['weight'],
                        delay=edge['delay'],
                        )
            new_complete_node = control_nodes[0]['complete']
            self.metadata['D'] = 0
            output_lists = [[self.name]]
        elif input_codings[0] == 'temporal-L':
            self.metadata['D'] = None
            new_complete_node = self.name+'_complete'
            graph.add_node(
                    new_complete_node,
                    index=-1,
                    threshold=len(input_lists[0]) - .00001,
                    decay=0.0,
                    p=1.0,
                    )
            output_neurons = []
            # Find 'begin' neuron -- We need to fix this
            # Tentatively is fixed!
            begin_neuron = control_nodes[0]['begin']
            # for input_neuron in [input_n for input_n in input_lists[0] if graph.nodes[input_n]['index']==-2]:
            #    begin_neuron = input_neuron

            for input_neuron in [input_n for input_n in input_lists[0] if graph.nodes[input_n]['index'] is not -2]:
                threshold_neuron_name = "{}_{}".format(self.name, graph.nodes[input_neuron]['index'])
                graph.add_node(
                        threshold_neuron_name,
                        index=graph.nodes[input_neuron]['index'],
                        threshold=1.0,
                        decay=0.0,
                        p=self.p,
                        )
                output_neurons.append(threshold_neuron_name)
                assert(type(self.threshold) is int)
                graph.add_edge(
                        begin_neuron,
                        threshold_neuron_name,
                        weight=2.0,
                        delay=self.threshold + 1,
                        )
                graph.add_edge(
                        input_neuron,
                        threshold_neuron_name,
                        weight=-3.0,
                        delay=1,
                        )
                graph.add_edge(
                        input_neuron,
                        new_complete_node,
                        weight=1.0,
                        delay=1,
                        )
                output_lists = [output_neurons]
            graph.add_edge(
                    begin_neuron,
                    new_complete_node,
                    weight=len(input_lists[0]),
                    delay=self.threshold + 1,
                    )
            graph.add_edge(
                    new_complete_node,
                    new_complete_node,
                    weight=-10,
                    delay=1,
                    )
        else:
            raise ValueError("Invalid coding")
        if self.output_coding is None:
            output_codings = ['unary-L']
        else:
            output_codings = [self.output_coding]
        self.is_built = True
        return (graph, self.metadata, [{'complete': new_complete_node}], output_lists, output_codings)
