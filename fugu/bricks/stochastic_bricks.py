#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from .bricks import Brick, input_coding_types


class PRN(Brick):
    """
    Psuedo-random neuron brick.
    Generates spikes randomly (a uniform random [0,1] draw is compared against a threshold).
    """
    def __init__(self,
                 probability=0.5,
                 steps=None,
                 shape=(1, ),
                 name="PRN",
                 output_coding='Undefined'):
        """
        Constructor for this brick.
        Args:
            probability: Probability of a spike at any timestep
            steps: Number of timesteps to produce spikes. None provides un-ending output.
            shape: shape of the neurons in the brick
            output_coding: Desired output coding for the brick
        """
        super(PRN, self).__init__(name)
        self.is_built = False
        self.metadata = {}
        self.probability = probability
        self.name = name
        self.shape = shape
        self.steps = steps
        self.output_coding = output_coding
        self.supported_codings = input_coding_types

    def build(self, graph, metadata, control_nodes, input_lists,
              input_codings):
        """
        Build Register brick.

        Args:
            graph: networkx graph to define connections of the computational graph
            metadata (dict): dictionary to define the shapes and parameters of the brick
            control_nodes (dict): dictionary of lists of auxillary networkx nodes.
                Expected keys:
                    'complete' - A list of neurons that fire when the brick is done
            input_lists (list): list of nodes that will contain input
            input_coding (list): list of input coding formats.  All coding types supported

        Returns:
            graph: graph of a computational elements and connections
            self.metadata: dictionary of output parameters (shape, coding, layers, depth, etc)
            complete (str): dictionary of control nodes ('complete')
            output_list (list): list of output
            self.output_coding (list): list of coding formats of output
        Raises:
              ValueError: PRN brick requires at least 1 input
        """
        if len(input_lists) == 0:
            raise ValueError("PRN brick requires at least 1 input.")
        # Driver Neuron
        driver_neuron = self.generate_neuron_name('driver')
        graph.add_node(driver_neuron, threshold=0.7, decay=1.0)
        graph.add_edge(driver_neuron, driver_neuron, weight=1.0, delay=1)
        # PRNs
        output_list = []
        for neuron_index in np.ndindex(self.shape):
            output_neuron = self.generate_neuron_name(str(neuron_index))
            graph.add_node(output_neuron,
                           threshold=0.7,
                           decay=1.0,
                           p=self.probability)
            output_list.append(output_neuron)
            graph.add_edge(driver_neuron, output_neuron, weight=1.0, delay=1)
        complete_neuron = self.generate_neuron_name('_complete')
        complete_threshold = self.steps - 1.1 if self.steps is not None else 1.0
        graph.add_node(complete_neuron,
                       threshold=complete_threshold,
                       decay=0.0)
        if self.steps is not None:
            graph.add_edge(driver_neuron, complete_neuron, weight=1.0, delay=1)
            graph.add_edge(complete_neuron,
                           driver_neuron,
                           weight=-10.0,
                           delay=1)
        for input_control_nodes in control_nodes:
            graph.add_edge(input_control_nodes['complete'],
                           driver_neuron,
                           weight=1.0,
                           delay=1)
        self.is_built = True
        return (graph, self.metadata, [{
            'complete': complete_neuron
        }], [output_list], [self.output_coding])


class Threshold(Brick):
    """
    Class to handle Threshold Brick. Inherits from Brick
    """
    def __init__(self,
                 threshold,
                 decay=0.0,
                 p=1.0,
                 name="Threshold",
                 output_coding=None):
        """
        Construtor for this brick.
        Args:
            threshold: Threshold value.  For input coding 'current', float.  For 'temporal-L', int.
            decay (float): Decay value for threshold neuron ('current' input only)
            p (float): Probability of firing when exceeding threshold ('current' input only)
            name (str): Name of the brick.  If not specified, a default will be used.  Name should be unique.
            output_coding: Force a return of this output coding.  Default is 'unary-L'
        """
        super(Threshold, self).__init__(name)
        self.is_built = False
        self.metadata = {}
        self.name = name
        self.p = p
        self.decay = decay
        self.threshold = threshold
        self.output_coding = output_coding
        self.supported_codings = ['current', 'Undefined', 'temporal-L']
        self.indices = []

    def set_properties(self, properties={}):
        if 'threshold' in properties:
            value = properties['threshold']
            neuron_props = {}
            if self.metadata['D'] == 0:
                neuron_props[self.generate_neuron_name("Main")] = {
                    'threshold': value
                }
            else:
                for index in self.indices:
                    name = self.generate_neuron_name("{}".format(index))
                    neuron_props[name] = {'threshold': value}
            return neuron_props, {}
        else:
            return None

    def build(self, graph, metadata, control_nodes, input_lists,
              input_codings):
        """
        Build Threshold brick.
        Args:
            graph: networkx graph to define connections of the computational graph
            metadata (dict): dictionary to define the shapes and parameters of the brick
            control_nodes (list): list of dictionary of auxillary nodes.
                  Expected keys:
                      'complete' - A neurons that fire when the brick is done
                      'begin' - A neurons that first when the brick begins processing (for temporal coded inputs)
            input_lists (list): list of nodes that will contain input
            input_coding (list): list of input coding formats

        Returns:
            graph: graph of a computational elements and connections
            self.metadata (dict): dictionary of output parameters (shape, coding, layers, depth, etc)
            new_complete_node: list dictionary of control nodes ('complete')
            output_lists (list[list[str]]): list of output
            output_codings (list[str]): list of coding formats of output
        """

        if len(input_codings) != 1:
            raise ValueError("Only one input is permitted.")
        if input_codings[0] not in self.supported_codings:
            raise ValueError(
                "Input coding not supported. Expected: {} ,Found: {}".format(
                    self.supported_codings,
                    input_codings[0],
                ))
        if input_codings[0] == 'current' or input_codings[0] == 'Undefined':
            main_neuron = self.generate_neuron_name("Main")
            graph.add_node(
                main_neuron,
                threshold=self.threshold,
                decay=self.decay,
                p=self.p,
            )
            for edge in input_lists[0]:
                graph.add_edge(
                    edge['source'],
                    main_neuron,
                    weight=edge['weight'],
                    delay=edge['delay'],
                )
            new_complete_node = control_nodes[0]['complete']
            self.metadata['D'] = 0
            output_lists = [[main_neuron]]
        elif input_codings[0] == 'temporal-L':
            self.metadata['D'] = None
            new_complete_node = self.name + '_complete'
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

            for input_neuron in [
                    input_n for input_n in input_lists[0]
                    if graph.nodes[input_n]['index'] != -2
            ]:
                index = graph.nodes[input_neuron]['index']
                self.indices.append(index)
                threshold_neuron_name = self.generate_neuron_name(
                    "{}".format(index))
                graph.add_node(
                    threshold_neuron_name,
                    index=index,
                    threshold=1.0,
                    decay=0.0,
                    p=self.p,
                )
                output_neurons.append(threshold_neuron_name)
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
        return (graph, self.metadata, [{
            'complete': new_complete_node
        }], output_lists, output_codings)
