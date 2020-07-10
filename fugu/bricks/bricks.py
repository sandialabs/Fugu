#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import abc
import sys

from abc import abstractmethod

if sys.version_info >= (3, 4):
    ABC = abc.ABC
else:
    ABC = abc.ABCMeta('ABC', (), {'__slots__': ()})

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
    return "{}{}".format(brick_name, Brick.brick_id)


class Brick(ABC):
    """
    Abstract Base Class definition of a Brick class
    """

    brick_id = 0

    def __init__(self, tag="Brick"):
        self.brick_tag = generate_brick_tag(tag)
        self.name = "Empty Brick"
        self.is_built = False
        self.supported_codings = []
        Brick.brick_id += 1
        
    def generate_neuron_name(self, neuron_name):
        """
        Adds the brick_tag to a neuron's name
        """
        return "{}:{}".format(self.brick_tag, neuron_name)

    @abstractmethod
    def build(self, graph, metadata, control_nodes, input_lists, input_codings):
        """
        Build the computational graph of the brick. Method must be defined in any class inheriting from Brick.

        Arguments:
            + graph - networkx graph
            + metadata - A dictionary of shapes and properties
            + control_nodes - list of dictionary of auxillary nodes.
                Acceptable keys include:
                    'complete' - A list of neurons that fire when the brick is done
                    'begin' - A list of neurons that fire when the brick begins computation
                                (used for temporal processing)
            + input_lists - list of lists of nodes for input neurons
            + input_codings - list of input coding types (as strings)
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

    def __init__(self, tag="InputBrick"):
        super(InputBrick, self).__init__(tag)
        self.name = "Input Brick"
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

        Arguments:
            + t - type of input (Default: None)
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

    def __init__(self, tag="CompoundBrick"):
        super(CompoundBrick, self).__init__(tag)
        self.name = "Compound Brick"
        self.children = {}

    def build_child(self, brick, graph, metadata, control_nodes, input_lists, input_codings):
        brick.brick_tag = self.brick_tag + ":" + brick.brick_tag
        self.children[brick.brick_tag] = brick
        return brick.build(graph, metadata, control_nodes, input_lists, input_codings)

    @abstractmethod
    def build(self, graph, metadata, control_nodes, input_lists, input_codings):
        """
        Build the computational graph of the brick. Method must be defined in any class inheriting from Brick.

        Arguments:
            + graph - networkx graph
            + metadata - A dictionary of shapes and properties
            + control_nodes - list of dictionary of auxillary nodes.
                Acceptable keys include:
                    'complete' - A list of neurons that fire when the brick is done
                    'begin' - A list of neurons that fire when the brick begins computation
                                (used for temporal processing)
            + input_lists - list of lists of nodes for input neurons
            + input_codings - list of input coding types (as strings)
        """
        pass

    def set_properties(self, graph, properties):
        """
        Returns an updated version of the graph based on the property values passed.
        """
        pass
