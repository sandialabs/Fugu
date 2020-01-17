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


def CalculateSpikeTimes(circuit, main_key='timestep'):
    initial_spikes = {}
    input_nodes = [node for node in circuit.nodes if ('layer' in circuit.nodes[node])
                   and (circuit.nodes[node]['layer'] == 'input')]
    max_steps = 0
    for input_node in input_nodes:
        for timestep, spike_list in enumerate(circuit.nodes[input_node]['brick']):
            if timestep > max_steps:
                max_steps = timestep
    if main_key == 'timestep':
        for i in range(0, max_steps + 1):
            initial_spikes[i] = deque()
        for input_node in input_nodes:
            for timestep, spike_list in enumerate(circuit.nodes[input_node]['brick']):
                if len(spike_list) > 0:
                    initial_spikes[timestep].extend(spike_list)
    elif main_key == 'neuron_name':
        for input_node in input_nodes:
            for timestep, spike_list in enumerate(circuit.nodes[input_node]['brick']):
                for neuron in spike_list:
                    if neuron not in initial_spikes:
                        initial_spikes[neuron] = []
                    initial_spikes[neuron].append(timestep)
    else:
        raise ValueError("main_key argument must be 'timestep' or 'neuron_name', not {}".format(main_key))

    return initial_spikes


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

        # @NOTE: Currently, this function behaves differently for Input Bricks
        #   - Instead of returning the changes, they change internally and reset the iterator
        #   - This is because of how initial spike times are calculated using said bricks
        #   - I have not yet found a way of incorporating my proposed method (above) into these bricks yet
        pass

    @abstractmethod
    def set_input_spikes(self):
        pass
