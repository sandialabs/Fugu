#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
isort:skip_file
"""

# fmt: off
import sys
from collections import deque
from warnings import warn

import fugu.simulators.SpikingNeuralNetwork as snn

from .backend import Backend, PortDataIterator
from ..utils.export_utils import results_df_from_dict
from ..utils.misc import CalculateSpikeTimes


class snn_Backend(Backend):
    def _build_network(self):
        self.nn = snn.NeuralNetwork()
        neuron_dict = {}
        """
        Add Neurons
        """

        # Add input neurons, as identified by circuit information.
        recordAll =  self.record == 'all'
        for node, vals in self.fugu_circuit.nodes.data():
            if vals.get('layer') != 'input': continue
            for neuron in PortDataIterator(vals):
                n = snn.InputNeuron(neuron, record=recordAll)
                neuron_dict[neuron] = n
                self.nn.add_neuron(n)

        # Add all other neurons.
        for neuron, props in self.fugu_graph.nodes.data():
            if neuron in neuron_dict: continue
            Vinit   =       props.get('voltage',       0.0)
            Vspike  =       props.get('threshold',     1.0)
            Vreset  =       props.get('reset_voltage', 0.0)
            Vretain = 1.0 - props.get('decay',         0.0)
            Vbias   =       props.get('bias',          0.0)
            P       =       props.get('p',             1.0)
            if 'potential'        in props: Vinit   = props['potential']
            if 'leakage_constant' in props: Vretain = props['leakage_constant']
            n = snn.LIFNeuron(neuron, voltage=Vinit, threshold=Vspike, reset_voltage=Vreset, leakage_constant=Vretain, bias=Vbias, p=P, record=recordAll)
            neuron_dict[neuron] = n
            self.nn.add_neuron(n)

        # Tag output neurons based on circuit information.
        if not recordAll:
            for node, vals in self.fugu_circuit.nodes.data():
                if vals.get('layer') != 'output': continue
                for neuron in PortDataIterator(vals):
                    neuron_dict[neuron].record = True

        for n1, n2, props in self.fugu_graph.edges.data():
            delay  = int(props.get('delay',  1))
            weight =     props.get('weight', 1.0)
            learning_rule = props.get('learning_rule', None)
            mod_neuron_name = props.get('mod_neuron', None)
            learning_params = props.get('learning_params', None)
            if mod_neuron_name is not None:
                mod_neuron = neuron_dict[mod_neuron_name]
            else:
                mod_neuron = None
            syn = snn.LearningSynapse(neuron_dict[n1],
                                      neuron_dict[n2],
                                      delay=delay,
                                      weight=weight,
                                      learning_rule = learning_rule,
                                      mod_neuron = mod_neuron,
                                      learning_params = learning_params)
            self.nn.add_synapse(syn)

        del neuron_dict
        '''
        Set initial input values
        '''
        self.set_input_spikes()

        if self.debug_mode:
            self.nn.list_neurons()

    def compile(self, scaffold, compile_args={}):
        # creates neuron populations and synapses
        self.fugu_circuit    = scaffold.circuit
        self.fugu_graph      = scaffold.graph
        self.brick_to_number = scaffold.brick_to_number
        self.record          = compile_args.get('record',     False)
        self.ds_format       = compile_args.get('ds_format',  True)
        self.debug_mode      = compile_args.get('debug_mode', False)
        self._build_network()

    def run(self, n_steps=10, return_potentials=False):
        """
        Runs the Simulator
        * runs circuit for n_steps then returns data
        * if not None raise error
        Returns:
        bool: True if ds format is required and converst neuron names to numbers and returns dictionary, False returns dataframe

        """
        output = self.nn.run(n_steps=n_steps,
                             debug_mode=self.debug_mode,
                             record_potentials=return_potentials)

        if return_potentials:
            df, final_potentials = output
        else:
            df = output

        if self.debug_mode:
            print(df)

        res = {}

        if self.ds_format:
            numerical_cols = {}
            for c in df.columns:
                numerical_cols[c] = self.fugu_graph.nodes[c]['neuron_number']
            df = df.rename(index=int, columns=numerical_cols)

            for r in df.index:
                col_list = []
                for c in df.columns:
                    if self.debug_mode:
                        if df.loc[r][c][0] == 1:
                            col_list.append(c)
                    else:
                        if df.loc[r][c] == 1:
                            col_list.append(c)
                    res[r] = col_list
            spike_times = results_df_from_dict(res, 'time', 'neuron_number')
        else:
            spike_times = results_df_from_dict(df, 'time', 'neuron_number')

        if return_potentials:
            return spike_times, final_potentials
        else:
            return spike_times

    def cleanup(self):
        # Deletes/frees neurons and synapses
        pass

    def reset(self):
        # resets time-step to 0 and resets neuron/synapse properties
        self._build_network()

    def set_properties(self, properties={}):
        """Set properties for specific neurons and synapses
        Args:
            properties: dictionary of properties for bricks
        """
        for brick in properties:
            if brick != 'compile_args':
                brick_id = self.brick_to_number[brick]
                changes = self.fugu_circuit.nodes[brick_id]['brick'].set_properties(properties[brick])
                if changes:
                    neuron_props, synapse_props = changes
                    for neuron_name in self.nn.nrns:
                        if neuron_name in neuron_props:
                            new_props = neuron_props[neuron_name]
                            for prop in new_props:
                                value = new_props[prop]
                                if prop == 'threshold':
                                    self.nn.nrns[neuron_name].threshold = value
                                elif prop == 'decay':
                                    self.nn.nrns[
                                        neuron_name].leakage_constant = value
                                elif prop == 'potential':
                                    self.nn.nrns[neuron_name].voltage = value
                                elif prop == 'p':
                                    self.nn.nrns[neuron_name].p = value

                    for synapse in self.nn.synps:
                        pre, post = self.nn.synps[
                            synapse].pre_neuron.name, self.nn.synps[
                                synapse].post_neuron.name
                        weight, delay = self.nn.synps[
                            synapse].weight, self.nn.synps[synapse].delay
                        if (pre, post) in synapse_props:
                            new_props = synapse_props[(pre, post)]
                            if 'weight' in new_props:
                                weight = new_props['weight']
                            elif 'delay' in new_props:
                                delay = new_props['delay']
                        if pre in synapse_props:
                            new_props = synapse_props[pre]
                            if 'weight' in new_props:
                                weight = new_props['weight']
                            elif 'delay' in new_props:
                                delay = new_props['delay']

                        self.nn.synps[synapse].set_params(delay, weight)

        self.set_input_spikes()

    def set_input_spikes(self):
        """
        Get new initial spike times
        """
        initial_spike_data = CalculateSpikeTimes(self.fugu_circuit)

        for key, vals in self.fugu_circuit.nodes.data():
            if vals.get('layer') != 'input': continue

            ports = vals.get('ports')
            if not ports: continue
            port = next(iter(ports.values()), None)  # find first entry, which is generally the only entry
            if not port: continue
            data_channel = port.channels.get('data')
            if not data_channel: continue

            initial_spikes = {}
            for neuron in data_channel.neurons:
                initial_spikes[neuron] = []
            for timestep in initial_spike_data:
                for neuron in initial_spike_data[timestep]:
                    if neuron in initial_spikes:
                        initial_spikes[neuron].append(1)
                for neuron in initial_spikes:
                    if (len(initial_spikes[neuron]) < timestep + 1):
                        initial_spikes[neuron].append(0)
            for neuron in data_channel.neurons:
                if self.debug_mode: print(neuron, initial_spikes[neuron])
                self.nn.update_input_neuron(neuron, initial_spikes[neuron])
