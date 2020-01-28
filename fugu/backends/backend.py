#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
        # creates neuron populations and synapses
        pass

    @abstractmethod
    def run(self, n_steps=10, return_potential=False):
        # Runs circuit for n_steps then returns data
        # If not None raise error
        pass

    @abstractmethod
    def cleanup(self):
        # Deletes/frees neurons and synapses
        pass

    @abstractmethod
    def reset(self):
        # Resets time-step to 0 and resets neuron/synapse properties
        pass

    @abstractmethod
    def set_properties(self, properties={}):
        # Set properties for specific neurons and synapses
        # properties = dictionary of parameter for bricks

        # Example:
        #   for brick in properties:
        #       neuron_props, synapse_props = self.circuit[brick].get_changes(properties[brick])
        #       for neuron in neuron_props:
        #           set neuron properties
        #       for synapse in synapse_props:
        #           set synapse properties

        # @NOTE: Currently, this function behaves differently for Input Bricks
        #   - Instead of returning the changes, they change internally and reset the iterator
        #   - This is because of how initial spike times are calculated using said bricks
        #   - I have not yet found a way of incorporating my proposed method (above) into these bricks yet
        pass

    @abstractmethod
    def set_input_spikes(self):
        pass
