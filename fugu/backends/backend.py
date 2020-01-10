#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import deque
from abc import abstractmethod

import abc
import sys
if sys.version_info >= (3, 4):
    ABC = abc.ABC
else:
    ABC = abc.ABCMeta('ABC', (), {'__slots__': ()})


class Backend(ABC):
    def _get_initial_spike_times(self, circuit):
        initial_spikes = {}
        input_nodes = [node for node in circuit.nodes if ('layer' in circuit.nodes[node])
                       and (circuit.nodes[node]['layer'] == 'input')]
        max_steps = 0
        for input_node in input_nodes:
            for timestep, spike_list in enumerate(circuit.nodes[input_node]['brick']):
                if timestep > max_steps:
                    max_steps = timestep
        for i in range(0, max_steps + 1):
            initial_spikes[i] = deque()
        for input_node in input_nodes:
            for timestep, spike_list in enumerate(circuit.nodes[input_node]['brick']):
                if len(spike_list) > 0:
                    initial_spikes[timestep].extend(spike_list)

        return initial_spikes

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
    def set_parameters(self, parameters={}):
        # Set parameters for specific neurons and synapses
        # parameters = dictionary of parameter for bricks
        # if not parameters: raise error

        # Example:
        #   for brick in parameters:
        #       neuron_props, synapse_props = self.circuit[brick].get_changes(parameters[brick])
        #       for neuron in neuron_props:
        #           set neuron properties
        #       for synapse in synapse_props:
        #           set synapse properties
        pass

    @abstractmethod
    def set_input_spikes(self):
        pass
