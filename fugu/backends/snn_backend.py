#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from collections import deque
from warnings import warn

import fugu.simulators.SpikingNeuralNetwork as snn

from .backend import Backend
from ..utils.export_utils import results_df_from_dict


class snn_Backend(Backend):

    def _build_network(self):
        self.nn = snn.NeuralNetwork()
        neuron_dict = {}

        '''
        Add Neurons
        '''
        # Add in input and output neurons. Use the fugu_circuit information to identify input and output layers
        # For input nodes, create input neurons and identity and associate the appropriate inputs to it
        # for output neurons, obtain neuron parameters from fugu_graph and create LIFNeurons
        # Add neurons to spiking neural network
        for node, vals in self.fugu_circuit.nodes.data():
            if 'layer' in vals:
                if vals['layer'] == 'input':
                    for neuron in self.fugu_circuit.nodes[node]['output_lists'][0]:
                        rc = True if self.record == 'all' else False
                        neuron_dict[neuron] = snn.InputNeuron(neuron, record=rc)
                        self.nn.add_neuron(neuron_dict[neuron])
                if vals['layer'] == 'output':
                    for olist in self.fugu_circuit.nodes[node]['output_lists']:
                        for neuron in olist:
                            params = self.fugu_graph.nodes[neuron]
                            th = params.get('threshold', 0.0)
                            rv = params.get('reset_voltage', 0.0)
                            lk = params.get('leakage_constant', 1.0)
                            vol = params.get('voltage', 0.0)
                            prob = params.get('p', 1.0)
                            if 'potential' in params:
                                vol = params['potential']
                            if 'decay' in params:
                                lk = 1.0 - params['decay']
                            neuron_dict[neuron] = snn.LIFNeuron(
                                                        neuron,
                                                        threshold=th,
                                                        reset_voltage=rv,
                                                        leakage_constant=lk,
                                                        voltage=vol,
                                                        p=prob,
                                                        record=True,
                                                        )
                            self.nn.add_neuron(neuron_dict[neuron])
        # add other neurons from self.fugu_graph to spiking neural network
        # parse through the self.fugu_graph and if a neuron is not present in spiking neural network, add to it.
        for neuron, params in self.fugu_graph.nodes.data():
            if neuron not in neuron_dict.keys():
                th = params.get('threshold', 0.0)
                rv = params.get('reset_voltage', 0.0)
                lk = params.get('leakage_constant', 1.0)
                vol = params.get('voltage', 0.0)
                prob = params.get('p', 1.0)
                if 'potential' in params:
                    vol = params['potential']
                if 'decay' in params:
                    lk = 1.0 - params['decay']
                rc = True if self.record == 'all' else False
                neuron_dict[neuron] = snn.LIFNeuron(
                                            neuron,
                                            threshold=th,
                                            reset_voltage=rv,
                                            leakage_constant=lk,
                                            voltage=vol,
                                            p=prob,
                                            record=rc,
                                            )
                self.nn.add_neuron(neuron_dict[neuron])

        '''
        Add Synapses
        '''
        # add synapses from self.fugu_graph edge information
        for n1, n2, params in self.fugu_graph.edges.data():
            delay = int(params.get('delay', 1))
            wt = params.get('weight', 1.0)
            syn = snn.Synapse(neuron_dict[n1], neuron_dict[n2], delay=delay, weight=wt)
            self.nn.add_synapse(syn)

        del neuron_dict
        '''
        Set initial input values
        '''
        self.set_input_spikes()

    def compile(self, scaffold, compile_args={}):
        # creates neuron populations and synapses
        self.fugu_circuit = scaffold.circuit
        self.fugu_graph = scaffold.graph
        self.brick_to_number = scaffold.brick_to_number
        if 'record' in compile_args:
            self.record = compile_args['record']
        else:
            self.record = False
        if 'ds_format' in compile_args:
            self.ds_format = compile_args['ds_format']
        else:
            self.ds_format = True
        if 'debug_mode' in compile_args:
            self.debug_mode = compile_args['debug_mode']
        else:
            self.debug_mode = False

        self._build_network()

    def run(self, n_steps=10, return_potentials=False):
        # runs circuit for n_steps then returns data
        # if not None raise error

        ''' Run the Simulator '''
        output = self.nn.run(n_steps=n_steps, debug_mode=self.debug_mode, record_potentials=return_potentials)

        if return_potentials:
            df, final_potentials = output
        else:
            df = output

        if self.debug_mode:
            print(df)

        res = {}
        # if ds format is required, convert neuron names to numbers and return dictionary
        # else return dataframe
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
        for brick in parameters:
            if brick != 'compile_args':
                brick_id = self.brick_to_number[brick]
                changes = self.fugu_circuit.nodes[brick_id]['brick'].set_parameters(parameters[brick])
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
                                    self.nn.nrns[neuron_name].leakage_constant = value
                                elif prop == 'potential':
                                    self.nn.nrns[neuron_name].voltage = value
                                elif prop == 'p':
                                    self.nn.nrns[neuron_name].p = value

                    for synapse in self.nn.synps:
                        pre, post = self.nn.synps[synapse].pre_neuron.name, self.nn.synps[synapse].post_neuron.name
                        weight, delay = self.nn.synps[synapse].weight, self.nn.synps[synapse].delay
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
        # Get new initial spike times
        initial_spike_data = self._get_initial_spike_times(self.fugu_circuit)

        for node, vals in self.fugu_circuit.nodes.data():
            if 'layer' in vals:
                if vals['layer'] == 'input':
                    initial_spikes = {}
                    for neuron in self.fugu_circuit.nodes[node]['output_lists'][0]:
                        initial_spikes[neuron] = []
                    for timestep in initial_spike_data:
                        spike_list = initial_spike_data[timestep]
                        for neuron in spike_list:
                            if neuron in initial_spikes:
                                initial_spikes[neuron].append(1)
                        for neuron in initial_spikes:
                            if(len(initial_spikes[neuron]) < timestep+1):
                                initial_spikes[neuron].append(0)

                    for neuron in self.fugu_circuit.nodes[node]['output_lists'][0]:
                        if self.debug_mode:
                            print(neuron, initial_spikes[neuron])
                        self.nn.update_input_neuron(neuron, initial_spikes[neuron])
