#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
These bricks are used to test features/capabilities of fugu backends.
"""
from .bricks import Brick, CompoundBrick, input_coding_types
from .register_bricks import Max, Addition


class Delay(Brick):
    """
    A brick that implements a simple chain of neurons.
    """
    def __init__(self, delay_value, name="Delay"):
        """
        Construtor for this brick.
        Args:
            name (str): Name of the brick.  If not specified, a default will be used.  Name should be unique.
        """
        super(Delay, self).__init__(name)
        self.is_built = False
        self.metadata = {'D': 1}
        self.name = name
        self.delay_value = delay_value

    def build(self, graph, metadata, control_nodes, input_lists,
              input_codings):
        """
        Build neuron chain brick.

        Args:
            graph ({add_node, add_edge}): networkx graph to define connections of the computational graph
            metadata (dict): dictionary to define the shapes and parameters of the brick
            control_nodes (list): list of dictionary of auxillary nodes.
                Expected keys:
                    'complete' - A list of neurons that fire when the brick is done
            input_lists (list): list of nodes that will contain input
            input_coding (list): list of input coding formats.  ('Raster', 'Undefined' supported)

        Returns:
            graph (any): graph of a computational elements and connections
            metadata (dict): dictionary of output parameters (shape, coding, layers, depth, etc)
            complete_name (str): list dictionary of control nodes ('complete')
            main_name: list of output edges
            input_codings (list): list of coding formats of output ('current')
        """

        num_inputs = sum([len(in_list) for in_list in input_lists])
        if num_inputs > 1:
            raise ValueError(
                "Input length does not match expected number. Expected: 1, Found: {}"
                .format(
                    num_inputs,
                ))

        graph.add_node(
            self.generate_neuron_name("begin"),
            threshold=0.5,
            potential=1.0,
            decay=0.0,
            index=-1,
            p=1.0,
        )
        complete_name = self.generate_neuron_name("complete")
        graph.add_node(
            complete_name,
            threshold=0.5,
            potential=0.0,
            decay=0.0,
            index=-1,
            p=1.0,
        )

        src_name = self.generate_neuron_name('src')
        dest_name = self.generate_neuron_name('dest')

        graph.add_node(
                src_name,
                threshold=0.1,
                potential=0.0,
                decay=0.0,
                index=0,
                )
        graph.add_node(
                dest_name,
                threshold=0.1,
                potential=0.0,
                decay=0.0,
                index=0,
                )

        graph.add_edge(
            input_lists[0][0],
            src_name,
            weight=1.0,
            delay=1,
            )

        graph.add_edge(
            src_name,
            dest_name,
            weight=1.0,
            delay=self.delay_value,
            )

        self.is_built = True
        return (graph, metadata, [{
            'complete': complete_name
        }], [[src_name, dest_name]], input_codings)


class NeuronChain(Brick):
    """
    A brick that implements a simple chain of neurons.
    """
    def __init__(self, num_neurons, name="NeuronChain"):
        """
        Construtor for this brick.
        Args:
            name (str): Name of the brick.  If not specified, a default will be used.  Name should be unique.
        """
        super(NeuronChain, self).__init__(name)
        self.is_built = False
        self.metadata = {'D': 1}
        self.name = name
        self.num_neurons = num_neurons

    def build(self, graph, metadata, control_nodes, input_lists,
              input_codings):
        """
        Build neuron chain brick.

        Args:
            graph ({add_node, add_edge}): networkx graph to define connections of the computational graph
            metadata (dict): dictionary to define the shapes and parameters of the brick
            control_nodes (list): list of dictionary of auxillary nodes.
                Expected keys:
                    'complete' - A list of neurons that fire when the brick is done
            input_lists (list): list of nodes that will contain input
            input_coding (list): list of input coding formats.  ('Raster', 'Undefined' supported)

        Returns:
            graph (any): graph of a computational elements and connections
            metadata (dict): dictionary of output parameters (shape, coding, layers, depth, etc)
            complete_name (str): list dictionary of control nodes ('complete')
            main_name: list of output edges
            input_codings (list): list of coding formats of output ('current')
        """

        num_inputs = sum([len(in_list) for in_list in input_lists])
        if num_inputs > 1:
            raise ValueError(
                "Input length does not match expected number. Expected: 1, Found: {}"
                .format(
                    num_inputs,
                ))

        graph.add_node(
            self.generate_neuron_name("begin"),
            threshold=0.5,
            potential=1.0,
            decay=0.0,
            index=-1,
            p=1.0,
        )
        complete_name = self.generate_neuron_name("complete")
        graph.add_node(
            complete_name,
            threshold=0.5,
            potential=0.0,
            decay=0.0,
            index=-1,
            p=1.0,
        )

        for i in range(self.num_neurons):
            neuron_name = self.generate_neuron_name(i)
            graph.add_node(
                    neuron_name,
                    threshold=0.1,
                    potential=0.0,
                    decay=0.0,
                    index=0,
                    )

        dest_name = neuron_name

        for i in range(self.num_neurons - 1):
            source_name = self.generate_neuron_name(i)
            dest_name = self.generate_neuron_name(i + 1)
            graph.add_edge(source_name, dest_name, weight=1.0, delay=1)

        graph.add_edge(
            input_lists[0][0],
            self.generate_neuron_name(0),
            weight=1.0,
            delay=1,
            )

        self.is_built = True
        return (graph, metadata, [{
            'complete': complete_name
        }], [[dest_name]], input_codings)


class InstantDecay(Brick):
    """
    A brick used to test neurons that have instant decay.
    """
    def __init__(self, num_inputs, name="InstantDecay"):
        """
        Construtor for this brick.
        Args:
            name (str): Name of the brick.  If not specified, a default will be used.  Name should be unique.
        """
        super(InstantDecay, self).__init__(name)
        self.is_built = False
        self.metadata = {'D': 1}
        self.name = name
        self.num_inputs = num_inputs

    def build(self, graph, metadata, control_nodes, input_lists,
              input_codings):
        """
        Build Dot brick.

        Args:
            graph ({add_node, add_edge}): networkx graph to define connections of the computational graph
            metadata (dict): dictionary to define the shapes and parameters of the brick
            control_nodes (list): list of dictionary of auxillary nodes.
                Expected keys:
                    'complete' - A list of neurons that fire when the brick is done
            input_lists (list): list of nodes that will contain input
            input_coding (list): list of input coding formats.  ('Raster', 'Undefined' supported)

        Returns:
            graph (any): graph of a computational elements and connections
            metadata (dict): dictionary of output parameters (shape, coding, layers, depth, etc)
            complete_name (str): list dictionary of control nodes ('complete')
            main_name: list of output edges
            input_codings (list): list of coding formats of output ('current')
        """

        num_inputs = sum([len(in_list) for in_list in input_lists])
        if num_inputs != self.num_inputs:
            raise ValueError(
                "Input length does not match expected number. Expected: {}, Found: {}"
                .format(
                    self.num_inputs,
                    num_inputs,
                ))

        graph.add_node(
            self.generate_neuron_name("begin"),
            threshold=0.5,
            potential=1.0,
            decay=0.0,
            index=-1,
            p=1.0,
        )
        complete_name = self.generate_neuron_name("complete")
        graph.add_node(
            complete_name,
            threshold=0.5,
            potential=0.0,
            decay=0.0,
            index=-1,
            p=1.0,
        )

        main_name = self.generate_neuron_name("main")
        graph.add_node(
            main_name,
            threshold=self.num_inputs - 0.01,
            potential=0.0,
            decay=1.0,
            index=0,
        )
        graph.add_edge(main_name, complete_name, weight=1.0, delay=1)

        for input_list in input_lists:
            for input_neuron in input_list:
                graph.add_edge(
                    input_neuron,
                    main_name,
                    weight=1.0,
                    delay=1,
                )

        self.is_built = True
        return (graph, metadata, [{
            'complete': complete_name
        }], [[main_name]], input_codings)


class SynapseProperties(Brick):
    """
    A brick used to test neurons that have instant decay.
    """
    def __init__(self, weights, name="SynapseProperties"):
        """
        Construtor for this brick.
        Args:
            name (str): Name of the brick.  If not specified, a default will be used.  Name should be unique.
        """
        super(SynapseProperties, self).__init__(name)
        self.is_built = False
        self.metadata = {'D': 1}
        self.name = name
        self.weights = weights

    def set_properties(self, properties):
        if 'weights' in properties:
            weights = properties['weights']
            if len(weights) != len(self.weights):
                raise ValueError(
                    "# of new weights ({}) != # of old weights ({})".format(
                        len(weights),
                        len(self.weights),
                    ))
            else:
                synapse_props = {}
                main_name = self.generate_neuron_name("main")
                for i, weight in enumerate(weights):
                    name = self.generate_neuron_name("{}".format(i))
                    synapse_props[(main_name, name)] = {'weight': weight}
                return {}, synapse_props

    def build(self, graph, metadata, control_nodes, input_lists,
              input_codings):
        """
        Build Dot brick.
        Args:
            graph: networkx graph to define connections of the computational graph
            metadata (dict): dictionary to define the shapes and parameters of the brick
            control_nodes (list): list of dictionary of auxillary nodes.
                Expected keys:
                    'complete' - A list of neurons that fire when the brick is done
            input_lists (list): list of nodes that will contain input
            input_coding (list): list of input coding formats.  ('Raster', 'Undefined' supported)

        Returns:
            graph: graph of a computational elements and connections
            metadata (dict): dictionary of output parameters (shape, coding, layers, depth, etc)
            complete_name (str): list dictionary of control nodes ('complete')
            output_list (list[str]): list of output edges
            input_coding (list): list of coding formats of output ('current')
        """

        begin_name = self.generate_neuron_name("begin")
        graph.add_node(
            begin_name,
            threshold=0.5,
            potential=1.0,
            decay=0.0,
            index=-1,
            p=1.0,
        )
        complete_name = self.generate_neuron_name("complete")
        graph.add_node(
            complete_name,
            threshold=0.5,
            potential=0.0,
            decay=0.0,
            index=-1,
            p=1.0,
        )

        main_name = self.generate_neuron_name("main")
        graph.add_node(
            main_name,
            threshold=0.5,
            potential=1.0,
            decay=0.0,
            index=-1,
            p=1.0,
        )

        output_list = []
        for index, weight in enumerate(self.weights):
            name = self.generate_neuron_name("{}".format(index))
            graph.add_node(
                name,
                threshold=1.0,
                potential=0.1,
                decay=1.0,
                index=-1,
            )
            graph.add_edge(
                main_name,
                name,
                weight=weight,
                delay=1,
            )
            output_list.append(name)

        self.is_built = True
        return (graph, metadata, [{
            'complete': complete_name
        }], [output_list], input_codings)


class SumOfMaxes(CompoundBrick):
    """
    A brick that computes the max of N sets of values and reports the sum of those maxes.
    """
    def __init__(self, set_sizes=[5, 5], name="SumOfMaxes"):
        """
        Construtor for this brick.
        Args:
            set_sizes (list[int]): Number of inputs for each max brick
            name (str): Name of the brick.  If not specified, a default will be used.  Name should be unique.
        """
        super(SumOfMaxes, self).__init__(name)
        self.is_built = False
        self.metadata = {'D': 1}
        self.name = name

        self.set_sizes = set_sizes

    def build(self, graph, metadata, control_nodes, input_lists,
              input_codings):
        """
        Build SumOfMaxes brick.

        Args:
            graph: networkx graph to define connections of the computational graph
            metadata (dict): dictionary to define the shapes and parameters of the brick
            control_nodes (list): list of dictionary of auxillary nodes.
                Expected keys:
                    'complete' - A list of neurons that fire when the brick is done
            input_lists (list): list of nodes that will contain input
            input_coding (list): list of input coding formats.  ('Raster', 'Undefined' supported)

        Returns:
            graph: graph of a computational elements and connections
            metadata (str): dictionary of output parameters (shape, coding, layers, depth, etc)
            complete_name (str): list dictionary of control nodes ('complete')
            child_output (any): list of output edges
            input_codings (list): list of coding formats of output ('current')
        Raises:
            ValueError: unexpected amount of input
        """

        if len(input_lists) != sum(self.set_sizes):
            raise ValueError(
                "Received an incorrect amount of input values. Expected {}, got {}"
                .format(
                    sum(self.set_sizes),
                    len(input_lists),
                ))

        begin_name = self.generate_neuron_name("begin")
        graph.add_node(
            begin_name,
            threshold=0.5,
            potential=1.0,
            decay=0.0,
            index=-1,
            p=1.0,
        )
        complete_name = self.generate_neuron_name("complete")
        graph.add_node(
            complete_name,
            threshold=0.5,
            potential=0.0,
            decay=0.0,
            index=-1,
            p=1.0,
        )

        max_base = "Max{}"
        index = 0

        default_max_size = max([len(i) for i in input_lists])

        addition_inputs = []
        addition_size = 0
        for i, set_size in enumerate(self.set_sizes):
            max_name = self.generate_neuron_name(max_base.format(i))
            max_inputs = []
            for j in range(set_size):
                value_neurons = []
                for neuron in input_lists[index]:
                    value_neurons.append(neuron)
                max_inputs.append(value_neurons)
                index += 1

            graph, _, _, child_output, _ = self.build_child(
                Max(default_size=default_max_size, name="Set{}".format(i)),
                graph,
                {},
                {},
                max_inputs,
                input_codings,
            )
            output_register = child_output[0]
            addition_inputs.append(output_register)
            if len(output_register) > addition_size:
                addition_size = len(output_register)

        graph, _, _, child_output, _ = self.build_child(
            Addition(register_size=addition_size + 1),
            graph,
            {},
            {},
            addition_inputs,
            input_codings,
        )

        self.is_built = True
        return (graph, metadata, [{
            'complete': complete_name
        }], child_output, input_codings)
