#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import abc
import sys

from abc import abstractmethod

if sys.version_info >= (3, 4):
    ABC = abc.ABC
else:
    ABC = abc.ABCMeta('ABC', (), {'__slots__': ()})

from ..scaffold import ChannelSpec, PortSpec, ChannelData, PortData, PortUtil

default_brick_metadata = {
    'input_shape': [()],
    'output_shape': [()],
    'D': 0,
    'layer': 'output',
    'input_coding': 'unknown',
    'output_coding': 'unknown',
}

input_coding_types = [
    'current',
    'unary-B',
    'unary-L',
    'binary-B',
    'binary-L',
    'temporal-B',
    'temporal-L',
    'Raster',
    'Population',
    'Rate',
    'Undefined',
]


def generate_brick_tag(brick_name):
    """
    Function that generates a unique brick tag
    """
    return "{}-{}".format(brick_name, Brick.brick_id)


class Brick(ABC):
    """
    Abstract Base Class definition of a Brick class
    """

    brick_id = 0

    def __init__(self, name="Brick"):
        self.brick_tag = generate_brick_tag(name)
        self.name = name
        self.is_built = False
        self.supported_codings = []
        Brick.brick_id += 1

    @classmethod
    def input_ports(cls) -> dict[str, PortSpec]:
        """
        Describes the ports on which this brick will take in data from other bricks.
        Returns a dictionary of PortSpec objects. The key is the name of the input
        port as known to this brick. See scaffold.py for the definition of PortSpec.
        A return value of {} indicate that this brick does not take inputs. Effectively, that
        flags this as an input brick, in the sense that all values originate from here.
        """
        return {}

    @classmethod
    def output_ports(cls) -> dict[str, PortSpec]:
        """
        Describes the ports on which this brick will output data to other bricks.
        Returns a dictionary of PortSpec objects. The key is the name of the output
        port as known to this brick. See scaffold.py for the definition of PortSpec.
        A return value of {} indicates that this brick does not produce outputs. This can happen
        if the brick does some form of direct I/O.
        """
        return {}

    @classmethod
    def show_ports(cls):
        ports = cls.input_ports()
        if not ports:
            print('No inputs')
        else:
            print('Inputs:')
            for port in ports.values(): cls.show_port(port, False)
        ports = cls.output_ports()
        if not ports:
            print('No outputs')
        else:
            print('Outputs:')
            for port in ports.values(): cls.show_port(port, True)

    @classmethod
    def show_port(cls, port: PortSpec, output: bool):
        """
        Subroutine of show_ports().
        """
        print(f"  '{port.name}'")
        if port.description: print('   ', port.description)
        print('    index   =', port.index)
        if not output:
            print('    minimum =', port.minimum)
            print('    maximum =', port.maximum)
        print('    channels:')
        for channel in port.channels.values():
            print(f"      '{channel.name}'")
            if not output:          print('       ', 'Required' if channel.required else 'Optional')
            if channel.description: print('       ', channel.description)
            if channel.coding:      print('        coding =', channel.coding)
            if channel.shape:       print('        shape  =', channel.shape)

    def generate_neuron_name(self, neuron_name):
        """
        Adds the brick_tag to a neuron's name
        """
        return "{}:{}".format(self.brick_tag, neuron_name)

    def build2(self, graph, inputs: dict[str, PortData] = {}):
        """
        Builds the computational graph of the brick.
        This method uses Ports to convey its inputs and outputs.

        The default implementation here acts as a shim to legacy bricks
        that use an older style of conveying inputs and outputs.
        New brick classes should override this method, and should avoid
        build() in any form.

        Args:
            graph (NetworkX.DiGraph): The neural network being built.
            inputs (dict): A collection ports feeding into this brick.
                The dictionery represents bindings between our input ports
                and the outputs supplied by other bricks. The key is the name
                of the input port as known to this brick and described by input_ports().
                The value is a PortData object created by the source brick.

        Returns:
            A dictionary of PortData objects. The key is the name of the port as
            known to this brick and described by output_ports().
        """
        metadata      = []
        control_nodes = []
        input_lists   = []
        input_codings = []
        ports = self.input_ports()  # Usually this will be {} for legacy bricks.
        for to_key, port_data in inputs.items():
            to_index = PortUtil.find_port_index(ports, to_key)  # Usually this will just be int(to_key).
            # ensure to_index position exists
            while to_index >= len(control_nodes):
                metadata     .append({})
                control_nodes.append({})
                input_lists  .append([])
                input_codings.append('Undefined')
            data_channel = port_data.channels.get('data')
            if data_channel:
                input_lists[to_index] = data_channel.neurons
                coding = data_channel.spec.coding
                if coding: input_codings[to_index] = coding[0] if type(coding) is list else coding
                shape = data_channel.spec.shape
                if shape: metadata[to_index]['output_shape'] = shape
            control_node = control_nodes[to_index]
            for name, channel_data in port_data.channels.items():
                if name=='data': continue
                neurons = channel_data.neurons
                control_node[name] = neurons[0] if len(neurons) == 1 else neurons

        (_, metadata, control_nodes, output_lists, output_codings) = self.build(graph, metadata, control_nodes, input_lists, input_codings)

        # Assemble output lists into ports
        outputs = {}
        if not type(metadata) is list:
            metadata = [metadata] * len(output_lists)
        for i in range(len(output_lists)):
            port_spec = PortSpec(name=str(i), index=i)
            port_data = PortData(spec=port_spec)
            outputs[port_spec.name] = port_data
            channel_spec = ChannelSpec(name='data')
            if i < len(metadata):       channel_spec.shape  = metadata[i].get('output_shape')
            if i < len(output_codings): channel_spec.coding = [output_codings[i]]
            port_data.channels['data'] = ChannelData(spec=channel_spec, neurons=output_lists[i])
            if i < len(control_nodes):
                for name, neuron in control_nodes[i].items():
                    channel = ChannelSpec(name)
                    port_data.channels[name] = ChannelData(spec=channel, neurons=[neuron])
        return outputs

    def build(self, graph, metadata, control_nodes, input_lists,
              input_codings):
        """
        Deprecated method for building the computational graph of the brick.
        New brick subclasses should override build2() instead.

        Args:
            graph (graph):  networkx graph
            metadata (dictionary): A dictionary of shapes and properties
            control_nodes (list): list of dictionary of auxillary nodes.
                Acceptable keys include:
                    'complete' - A list of neurons that fire when the brick is done
                    'begin' - A list of neurons that fire when the brick begins computation
                                (used for temporal processing)
            input_lists (list): list of lists of nodes for input neurons
            input_codings (list): list of input coding types (as strings)
        """
        pass

    def set_properties(self, graph, properties):
        """
        Returns an updated version of the graph based on the property values passed.
        """
        pass


class InputBrick(Brick):
    """
    Abstract Base class for handling inputs inherited from Brick
    """
    def __init__(self, name="InputBrick"):
        super(InputBrick, self).__init__(name)
        self.name = name
        self.streaming = False

    @abstractmethod
    def __iter__(self):
        pass

    @abstractmethod
    def __next__(self):
        pass

    @abstractmethod
    def get_input_value(self, t=None):
        """
        Abstract method to get input values. InputBricks must implement this method

        Args:
            t: type of input (Default: None)
        """
        pass

    def set_properties(self, properties=None):
        pass


class CompoundBrick(Brick):
    """
    Abstract Base Class definition of a CompoundBrick class.
    A CompoundBrick is brick that contains other bricks.
    """

    brick_id = 0

    def __init__(self, name="CompoundBrick"):
        super(CompoundBrick, self).__init__(name)
        self.name = name
        self.children = {}

    def build_child(self, brick, graph, metadata, control_nodes, input_lists,
                    input_codings):
        brick.brick_tag = self.brick_tag + ":" + brick.brick_tag
        self.children[brick.brick_tag] = brick
        return brick.build(graph, metadata, control_nodes, input_lists,
                           input_codings)

    @abstractmethod
    def build(self, graph, metadata, control_nodes, input_lists,
              input_codings):
        """
        Build the computational graph of the brick. Method must be defined in any class inheriting from Brick.

        Args:
            graph: networkx graph
            metadata: A dictionary of shapes and properties
            control_nodes (list): list of dictionary of auxillary nodes.
                Acceptable keys include:
                    'complete' - A list of neurons that fire when the brick is done
                    'begin' - A list of neurons that fire when the brick begins computation
                                (used for temporal processing)
            input_lists (list): list of lists of nodes for input neurons
            input_codings (list): list of input coding types (as strings)
        """
        pass

    def set_properties(self, graph, properties):
        """
        Returns an updated version of the graph based on the property values passed.
        """
        pass
