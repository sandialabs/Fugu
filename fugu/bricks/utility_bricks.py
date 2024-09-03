#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from .bricks import Brick, input_coding_types
from ..scaffold import ChannelSpec, PortSpec, ChannelData, PortData, PortUtil


class Dot(Brick):
    """
    Class to handle the Dot brick. Inherits from Brick
    """
    def __init__(self, weights, name="Dot"):
        """
        Construtor for this brick.
        Args:
            weights (any): Vector against which the input is dotted.
            name (str): Name of the brick.  If not specified, a default will be used.  Name should be unique.
        """
        super(Dot, self).__init__(name)
        self.is_built = False
        self.metadata = {'D': 1}
        self.name = name
        self.weights = weights
        self.supported_codings = ['Raster', 'Undefined']
        self.input_sources = []

    def set_properties(self, properties):
        if 'weights' in properties:
            weights = properties['weights']
            if len(weights) != len(self.input_sources):
                raise ValueError(
                    "# of weights ({}) != # of inputs to this, {}, Dot brick ({})"
                    .format(
                        len(weights),
                        self.name,
                        len(self.input_sources),
                    ), )
            else:
                synapse_props = {}
                for neuron, weight in zip(self.input_sources, weights):
                    synapse_props[neuron] = {'weight': weight}
                return {}, synapse_props

    def build(self, graph, metadata, control_nodes, input_lists,
              input_codings):
        """
        Build Dot brick.

        Args:
            graph (any): networkx graph to define connections of the computational graph
            metadata (dict): dictionary to define the shapes and parameters of the brick
            control_nodes (list): list of dictionary of auxillary nodes.
                Expected keys:
                    'complete' - A list of neurons that fire when the brick is done
            input_lists (list): list of nodes that will contain input
            input_coding (list): list of input coding formats.  ('Raster', 'Undefined' supported)

        Returns:
            graph: graph of a computational elements and connections
            metadata: dictionary of output parameters (shape, coding, layers, depth, etc)
            complete_name: list dictionary of control nodes ('complete')
            output_list (list[dict[str, int]]): list of output edges
            output_codings (list[str]): list of coding formats of output ('current')
        """

        output_list = []
        output_codings = ['current']
        if len(input_codings) > 1:
            raise ValueError("Only one input is permitted.")
        if input_codings[0] not in self.supported_codings:
            raise ValueError(
                "Input coding not supported. Expected: {}, Found: {}".format(
                    self.supported_codings,
                    input_codings[0],
                ))
        if len(input_lists[0]) != len(self.weights):
            raise ValueError(
                "Input length does not match weights. Expected: {}, Found: {}".
                format(
                    len(self.weights),
                    len(input_lists[0]),
                ))
        for i, weight in enumerate(self.weights):
            output_list.append({
                'source': input_lists[0][i],
                'weight': weight,
                'delay': 1
            })
            self.input_sources.append(input_lists[0][i])
        if type(metadata) is list:
            metadata = metadata[0]
        #metadata['D'] = metadata['D'] + 1   # This method of timing is now deprectated.
        complete_name = self.generate_neuron_name("complete")
        graph.add_node(
            complete_name,
            threshold=0.5,
            potential=0.0,
            decay=0.0,
            index=-1,
            p=1.0,
        )
        graph.add_edge(
            control_nodes[0]['complete'],
            complete_name,
            weight=1.0,
            delay=1,
        )
        self.is_built = True
        return (graph, metadata, [{
            'complete': complete_name
        }], [output_list], output_codings)


class Copy(Brick):
    """
    Class to handle Copy Brick. Inherits from Brick
    """
    def __init__(self, name="Copy"):
        """
        Construtor for this brick.
        Args:
            name (str): Name of the brick.  If not specified, a default will be used.  Name should be unique.
        """
        super(Copy, self).__init__(name)
        self.is_built = False
        self.metadata = {'D': 1}
        self.name = name
        self.supported_codings = [
            'unary-B',
            'unary-L',
            'binary-B',
            'binary-L',
            'temporal-B',
            'temporal-L',
            'Raster',
            'Undefined',
        ]

    def build(self, graph, metadata, control_nodes, input_lists,
              input_codings):
        """
        Build Copy brick.

        Args:
            graph: networkx graph to define connections of the computational graph
            metadata (dict): dictionary to define the shapes and parameters of the brick
            control_nodes (list): list of dictionaries of auxillary nodes.
                Expected keys:
                    'complete' - A list of neurons that fire when the brick is done
            input_lists (list): list of nodes that will contain input
            input_coding (list): list of input coding formats

        Returns:
            graph: graph of a computational elements and connections
            self.metadata (dict): dictionary of output parameters (shape, coding, layers, depth, etc)
            complete: list dictionary of control nodes ('complete')
            output_lists (list[list]): list of output
            output_codings (list): list of coding formats of output
        """
        num_copies = 2
        if type(metadata) is list:
            self.metadata = metadata[0]
        else:
            self.metadata = metadata
        self.metadata['D'] = 1
        if len(input_lists) > 1:
            print('Copying first input only!')
        if input_codings[0] not in self.supported_codings:
            print("Unsupported input coding: " + input_codings[0])
            return -1
        output_lists = [[] for i in range(num_copies)]
        output_codings = []
        for neuron in input_lists[0]:
            for copy_num in range(0, num_copies):
                copy_name = self.generate_neuron_name("{}_copy{}".format(
                    neuron, copy_num))
                graph.add_node(copy_name,
                               threshold=0.5,
                               decay=0,
                               p=1.0,
                               index=graph.nodes[neuron]['index'])
                graph.add_edge(neuron, copy_name, weight=1.0, delay=1)
                output_lists[copy_num].append(copy_name)
        for copy_num in range(0, num_copies):
            output_codings.append(input_codings[0])
        complete_name = self.generate_neuron_name("complete")
        graph.add_node(
            complete_name,
            threshold=0.5,
            decay=0,
            p=1.0,
            index=-1,
        )
        graph.add_edge(
            control_nodes[0]['complete'],
            complete_name,
            weight=1.0,
            delay=1,
        )
        self.is_built = True
        return (
            graph,
            self.metadata,
            [{
                'complete': self.name + "_complete"
            }] * num_copies,
            output_lists,
            output_codings,
        )


class Concatenate(Brick):
    """
    Brick that concatenates multiple inputs into a single vector.
    All codings are supported except 'current'; first coding is used if not specified.

    """
    def __init__(self, name="Concatenate", coding=None):
        """
        Args:
            name (str): Name of the brick.  If not specified, a default will be used.  Name should be unique.
        """
        super(Concatenate, self).__init__(name)
        self.is_built = False
        self.metadata = {'D': 0}
        self.name = name
        self.supported_codings = input_coding_types
        if coding is not None:
            self.coding = coding
        else:
            self.coding = None

    def build(self, graph, metadata, control_nodes, input_lists,
              input_codings):
        """
        Build concatenate brick.

        Args:
            graph: networkx graph to define connections of the computational graph
            metadata (dict): dictionary to define the shapes and parameters of the brick
            control_nodes (list): dictionary of lists of auxillary networkx nodes.
                Expected keys:
                    'complete' - A list of neurons that fire when the brick is done
            input_lists (list): list of nodes that will contain input
            input_coding (list): list of input coding formats.  All codings are allowed except 'current'.

        Returns:
            graph of a computational elements and connections
            self.metadata (dict[str, int]): dictionary (dict[str, int]) of output parameters (shape, coding, layers, depth, etc)
            dictionary of control nodes ('complete')
            output_list: list of lists of output (1 output)
            output_codings (list): list of coding formats of output (Coding matches input coding)
        """
        # Keep the same coding as input 0 for the output
        # This is an arbitrary decision at this point.
        # Generally, your brick will impart some coding, but that isn't the case here.
        if self.coding is None:
            output_codings = [input_codings[0]]
        else:
            output_codings = [self.coding]

        complete_name = self.generate_neuron_name("complete")
        graph.add_node(
            complete_node_name,
            index=-1,
            threshold=1.0,
            decay=0.0,
            p=1.0,
            potential=0.0,
        )
        for idx in range(len(input_lists)):
            graph.add_edge(
                control_nodes[idx]['complete'],
                complete_node_name,
                weight=(1 / len(input_lists)) + 0.000001,
                delay=1,
            )

        output_lists = [[]]
        for input_brick in input_lists:
            for input_neuron in input_brick:
                relay_neuron_name = self.generate_neuron_name(
                    "relay_{}".format(input_neuron))
                graph.add_node(
                    relay_neuron_name,
                    index=(len(output_lists[0]), ),
                    threshold=0.0,
                    decay=0.0,
                    p=1.0,
                    potential=0.0,
                )
                graph.add_edge(input_neuron,
                               relay_neuron_name,
                               weight=1.0,
                               delay=1)
                output_lists[0].append(relay_neuron_name)

        self.is_built = True

        return (graph, self.metadata, [{
            'complete': complete_node_name
        }], output_lists, output_codings)


class AND_OR(Brick):
    """
    Brick for performing a logical AND/OR.
    Operation is performed entry-wise, matching based on index.  All codings are supported.

    """
    def __init__(self, mode='AND', name="AND_OR"):  # A change here
        """
        Args:
            mode (str): Either 'And' or 'Or'; determines the operation
            name (str): Name of the brick.  If not specified, a default will be used.  Name should be unique.
        """
        super(AND_OR, self).__init__(name)
        # The brick hasn't been built yet.
        self.is_built = False
        # We just store the name passed at construction.
        self.name = name
        self.mode = mode  # A change here

    @classmethod
    def input_ports(cls) -> dict[str, PortSpec]:
        port = PortSpec(name='input', minimum=2, maximum=2)
        port.channels['data']     = ChannelSpec(name='data', coding=input_coding_types)
        port.channels['complete'] = ChannelSpec(name='complete')
        return {port.name: port}

    @classmethod
    def output_ports(cls) -> dict[str, PortSpec]:
        port = PortSpec(name='output')
        port.channels['data']     = ChannelSpec(name='data', coding=input_coding_types)
        port.channels['complete'] = ChannelSpec(name='complete')
        return {port.name: port}

    def build2(self, graph, inputs: dict[str, PortData] = {}):
        """
        Build AND_OR brick.
        Raises:
            ValueError: If != 2 inputs.  Only 2 inputs are supported.  Error if unsupported mode.
        """
        # Expect two inputs
        if len(inputs) != 2:
            raise ValueError('Only two inputs supported.')
        # Only two supported modes, AND and OR
        if self.mode != 'AND' and self.mode != 'OR':
            raise ValueError('Unsupported mode.')

        input1, input2 = PortUtil.get_autoports(inputs, 'input', 2)
        result = PortUtil.make_ports_from_specs(AND_OR.output_ports())
        output = result['output']
        data = output.channels['data']
        # Keep the same coding as input 0 for the output
        # This is an arbitrary decision at this point.
        # Generally, your brick will impart some coding, but that isn't the case here.
        data.spec.coding = input1.channels['data'].spec.coding

        complete_node_name = self.generate_neuron_name('complete')
        output.channels['complete'].neurons = [complete_node_name]
        graph.add_node(complete_node_name,
                       index=-1,
                       threshold=0.0,
                       decay=0.0,
                       p=1.0,
                       potential=0.0)
        graph.add_edge(input1.channels['complete'].neurons[0],
                       complete_node_name,
                       weight=1.0,
                       delay=1)

        threshold_value = 1.0 if self.mode == 'AND' else 0.5
        # We also, obviously, need to build the computational portion of our graph
        data1 = input1.channels['data'].neurons
        data2 = input2.channels['data'].neurons
        for i in range(min(len(data1), len(data2))):
            operand1 = data1[i]
            operand2 = data2[i]
            # Remember all of our output neurons need to be marked
            and_node_name = self.generate_neuron_name(f"{operand1}_{operand2}")
            data.neurons.append(and_node_name)
            graph.add_node(and_node_name,
                           index=0,
                           threshold=threshold_value,
                           decay=1.0,
                           p=1.0,
                           potential=0.0)
            graph.add_edge(operand1,
                           and_node_name,
                           weight=0.75,
                           delay=1)
            graph.add_edge(operand2,
                           and_node_name,
                           weight=0.75,
                           delay=1)

        self.is_built = True
        return result


class ParityCheck(Brick):
    """
    Brick to compute the parity of a 4 bit input.
    The output spikes after 2 time steps if the input has odd parity
    """

    # author: Srideep Musuvathy
    # email: smusuva@sandia.gov
    # last updated: April 8, 2019'''

    def __init__(self, name="ParityCheck"):
        """
        Construtor for this brick.
        Args:
            name (str): Name of the brick.  If not specified, a default will be used.  Name should be unique.
        """
        super(ParityCheck, self).__init__(name)
        self.is_built = False
        self.metadata = {'D': 1}
        self.name = name
        self.supported_codings = ['binary-B', 'binary-L', 'Raster']

    def build(self, graph, metadata, control_nodes, input_lists,
              input_codings):
        """
        Build Parity brick.

        Args:
            graph: networkx graph to define connections of the computational graph
            metadata (dict): dictionary to define the shapes and parameters of the brick
            control_nodes (list): dictionary of lists of auxillary networkx nodes.
                Expected keys:
                    'complete' - A list of neurons that fire when the brick is done
            input_lists (list): list of nodes that will contain input
            input_coding (list): list of input coding formats
        Returns:
            graph: graph of a computational elements and connections
            self.metadata (dict): dictionary of output parameters (shape, coding, layers, depth, etc).  dict[str, int] = {'D': 1}
            complete_node (str): dictionary of control nodes ('complete')
            output_lists (list[list[str]]): list of output
            output_codings (list): list of coding formats of output
        Example:
            add 4 hidden nodes with thresholds <=1, >=1, <=3, >=3.
            since the thresholds only compute >=, the <=1, <=3 computations are performed by negating
            the threshold weights and the inputs (via the weights on incomming edges)
            first hidden node and connect edges from input layer
        """

        if len(input_codings) != 1:
            raise ValueError('Parity check takes in 1 input')

        output_codings = [input_codings[0]]

        complete_node_name = self.generate_neuron_name('complete')

        graph.add_node(
            complete_node_name,
            index=-1,
            threshold=0.0,
            decay=0.0,
            p=1.0,
            potential=0.0,
        )
        graph.add_edge(control_nodes[0]['complete'],
                       complete_node_name,
                       weight=1.0,
                       delay=2)
        complete_node = complete_node_name


        h_00 = self.generate_neuron_name('h_00')
        graph.add_node(
            h_00,
            index=0,
            threshold=-1.1,
            decay=1.0,
            p=1.0,
            potential=0.0,
        )
        graph.add_edge(
            input_lists[0][0],
            h_00,
            weight=-1.0,
            delay=1,
        )
        graph.add_edge(
            input_lists[0][1],
            h_00,
            weight=-1.0,
            delay=1,
        )
        graph.add_edge(
            input_lists[0][2],
            h_00,
            weight=-1.0,
            delay=1,
        )
        graph.add_edge(
            input_lists[0][3],
            h_00,
            weight=-1.0,
            delay=1,
        )

        # second hidden node and edges from input layer
        h_01 = self.generate_neuron_name('h_01')
        graph.add_node(
            h_01,
            index=1,
            threshold=0.9,
            decay=1.0,
            p=1.0,
            potential=0.0,
        )
        graph.add_edge(
            input_lists[0][0],
            h_01,
            weight=1.0,
            delay=1,
        )
        graph.add_edge(
            input_lists[0][1],
            h_01,
            weight=1.0,
            delay=1,
        )
        graph.add_edge(
            input_lists[0][2],
            h_01,
            weight=1.0,
            delay=1,
        )
        graph.add_edge(
            input_lists[0][3],
            h_01,
            weight=1.0,
            delay=1,
        )

        # third hidden node and edges from input layer
        h_02 = self.generate_neuron_name('h_02')
        graph.add_node(
            h_02,
            index=2,
            threshold=-3.1,
            decay=1.0,
            p=1.0,
            potential=0.0,
        )
        graph.add_edge(
            input_lists[0][0],
            h_02,
            weight=-1.0,
            delay=1,
        )
        graph.add_edge(
            input_lists[0][1],
            h_02,
            weight=-1.0,
            delay=1,
        )
        graph.add_edge(
            input_lists[0][2],
            h_02,
            weight=-1.0,
            delay=1,
        )
        graph.add_edge(
            input_lists[0][3],
            h_02,
            weight=-1.0,
            delay=1,
        )

        # fourth hidden node and edges from input layer
        h_03 = self.generate_neuron_name('h_03')
        graph.add_node(
            h_03,
            index=3,
            threshold=2.9,
            decay=1.0,
            p=1.0,
            potential=0.0,
        )
        graph.add_edge(
            input_lists[0][0],
            h_03,
            weight=1.0,
            delay=1,
        )
        graph.add_edge(
            input_lists[0][1],
            h_03,
            weight=1.0,
            delay=1,
        )
        graph.add_edge(
            input_lists[0][2],
            h_03,
            weight=1.0,
            delay=1,
        )
        graph.add_edge(
            input_lists[0][3],
            h_03,
            weight=1.0,
            delay=1,
        )

        # output_node and edges from hidden nodes
        parity = self.generate_neuron_name('parity')
        graph.add_node(
            parity,
            index=4,
            threshold=2.9,
            decay=1.0,
            p=1.0,
            potential=0.0,
        )
        graph.add_edge(
            h_00,
            parity,
            weight=1.0,
            delay=1,
        )
        graph.add_edge(
            h_01,
            parity,
            weight=1.0,
            delay=1,
        )
        graph.add_edge(
            h_02,
            parity,
            weight=1.0,
            delay=1,
        )
        graph.add_edge(
            h_03,
            parity,
            weight=1.0,
            delay=1,
        )

        self.is_built = True

        output_lists = [[parity]]

        return (graph, self.metadata, [{
            'complete': complete_node
        }], output_lists, output_codings)


class TemporalAdder(Brick):
    """
    Brick that "adds" spike times together:

    More specifically, consider you have three neurons u, v, and w that first spike at times t_u, t_v, and t_w.
    Assuming v spikes before w (so t_v < t_w), we want t_u = t_w + t_v, i.e. u fires t_v timesteps after w fires.
    """
    def __init__(self,
                 number_of_elements,
                 design='default',
                 name="TemporalAdder",
                 output_coding='temporal-L'):
        """
        Construtor for this brick.
        Args:
            number_of_elements (any): number of signals you want to add together
            design (str): default
            name (str): Name of the brick.  If not specified, a default will be used.  Name should be unique.
            output_coding (str): Output coding type, default is 'temporal-L'
        """
        super(TemporalAdder, self).__init__(name)
        self.is_built = False
        self.name = name
        self.supported_codings = input_coding_types

        self.output_codings = [output_coding]
        self.metadata = {'D': None}

        self.num_elements = number_of_elements
        self.design = design

    def build(self, graph, metadata, control_nodes, input_lists,
              input_codings):
        """
        Build Adder brick.

        Args:
            graph: networkx graph to define connections of the computational graph
            metadata: dictionary to define the shapes and parameters of the brick
            control_nodes: dictionary of lists of auxillary networkx nodes.
                Expected keys:
                    'complete' - A list of neurons that fire when the brick is done
                    'begin' - A neurons that first when the brick begins processing (for temporal coded inputs)
            input_lists: list of nodes that will contain input
            input_coding: list of input coding formats.  All coding types supported

        Returns:
            graph: graph of a computational elements and connections
            self.metadata: dictionary of output parameters (shape, coding, layers, depth, etc)
            complete_node_list: dictionary of control nodes ('complete')
            output_lists: list of output
            self.output_coings: list of coding formats of output
        Raises:
            ValueError: incorrect number or format of inputs

        """

        if len(input_lists) < 2:
            if len(input_lists) > 0:
                if len(input_lists[0]) < 2:
                    # @TODO: Figure out correct way of handling this case
                    pass
            else:
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
                       threshold=0.1,
                       decay=0.0,
                       potential=0.0)

        complete_name = self.generate_neuron_name('complete')
        graph.add_node(complete_name, threshold=0.1, decay=0.0, potential=0.0)
        complete_node_list = [complete_name]

        output_name = self.generate_neuron_name("Sum")
        graph.add_node(output_name, threshold=0.00, decay=0.0, potential=-0.01)

        graph.add_edge(output_name, complete_name, weight=1.0, delay=2)
        graph.add_edge(output_name, output_name, weight=-5.0, delay=2)

        increment_timer_name = self.generate_neuron_name("T_I")
        decrement_timer_name = self.generate_neuron_name("T_D")
        graph.add_node(increment_timer_name,
                       threshold=self.num_elements - 0.01,
                       decay=0.0,
                       potential=0.0)
        graph.add_edge(increment_timer_name,
                       increment_timer_name,
                       weight=self.num_elements,
                       delay=2)
        graph.add_edge(increment_timer_name,
                       output_name,
                       weight=1.0,
                       delay=2)
        graph.add_node(decrement_timer_name,
                       threshold=0.99,
                       decay=0.0,
                       potential=1.0)
        graph.add_edge(decrement_timer_name,
                       decrement_timer_name,
                       weight=1.0,
                       delay=2)
        graph.add_edge(decrement_timer_name,
                       output_name,
                       weight=-1.0,
                       delay=2)

        graph.add_edge(output_name,
                       increment_timer_name,
                       weight=-1 * self.num_elements,
                       delay=2)
        graph.add_edge(output_name,
                       decrement_timer_name,
                       weight=-1 * self.num_elements,
                       delay=2)

        for input_list in input_lists:
            for input_signal in input_list:
                graph.add_edge(input_signal,
                               increment_timer_name,
                               weight=1.0,
                               delay=2)
                graph.add_edge(input_signal,
                               decrement_timer_name,
                               weight=-2.0,
                               delay=2)

        self.is_built = True

        # Remember, bricks can have more than one output, so we need a list of
        # list of output neurons
        output_lists = [[output_name]]

        return (
            graph,
            self.metadata,
            [{
                'complete': complete_node_list[0],
                'begin': begin_node_name
            }],
            output_lists,
            self.output_codings,
        )
