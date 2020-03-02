#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 14:46:55 2019

@author: smusuva
"""
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
