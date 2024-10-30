#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
isort:skip_file
"""

# fmt: off
from abc import abstractmethod

import abc
import sys
if sys.version_info >= (3, 4):
    ABC = abc.ABC
else:
    ABC = abc.ABCMeta('ABC', (), {'__slots__': ()})


class Backend(ABC):
    @abstractmethod
    def compile(self, scaffold, compile_args={}):
        """
        creates neuron populations and synapses
        """
        pass

    @abstractmethod
    def run(self, n_steps=10, return_potential=False):
        "Runs circuit for n_steps then returns data"
        "If not None raise error"
        pass

    @abstractmethod
    def cleanup(self):
        """
        Deletes/frees neurons and synapses
        """
        pass

    @abstractmethod
    def reset(self):
        """
        Resets time-step to 0 and resets neuron/synapse properties
        """
        pass

    @abstractmethod
    def set_properties(self, properties={}):
        """
        Set properties for specific neurons and synapses
        Args:
            properties: dictionary of parameter for bricks

        Example:
           for brick in properties:
               neuron_props, synapse_props = self.circuit[brick].get_changes(properties[brick])
               for neuron in neuron_props:
                   set neuron properties
               for synapse in synapse_props:
                   set synapse properties

        @NOTE: Currently, this function behaves differently for Input Bricks
           * Instead of returning the changes, they change internally and reset the iterator
           * This is because of how initial spike times are calculated using said bricks
           * I have not yet found a way of incorporating my proposed method (above) into these bricks yet
        """
        pass

    @abstractmethod
    def set_input_spikes(self):
        pass


class PortDataIterator:
    """
        Given the attribute dictionary associated with a circuit graph node,
        this iterator returns all the neurons that reside in any of its output
        port 'data' channels, one at a time.
    """

    def __init__(self, node: dict):
        self.neurons = None
        ports = node.get('ports')
        if ports:  # prepare top-level iterator
            self.ports = iter(ports.values())
            try:
                self.findNeurons()
            except StopIteration:
                self.ports = None

    def __getitem__(self, i):
        if not self.ports: raise StopIteration
        while True:  # This loop ends by either a successful return or StopIteration
            try:
                return self.neurons.__next__()
            except StopIteration:
                self.findNeurons()  # This returns StopIteration when the ports iterator is exhausted.

    def findNeurons(self):
        self.neurons = None
        while not self.neurons:
            port = self.ports.__next__()   # Can raise StopIeration directly
            data = port.channels.get('data')
            if data: self.neurons = iter(data.neurons)
        # At this point, self.neurons has a fresh iterator and self.ports is positioned just ahead of next port.
