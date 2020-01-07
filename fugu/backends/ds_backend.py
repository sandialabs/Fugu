#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import networkx as nx

from .backend import Backend
from fugu.simulators.ds import run_simulation


class ds_Backend(Backend):
    def __init__(self):
        super(Backend, self).__init__()
        self.scaffold = None
        self.ds_graph = None
        self.neuron_to_id_map = {}

    def _create_ds_injection(self, input_values):
        # find input nodes
        import torch

        injection_tensors = {}
        for t in input_values:
            injection_tensors[t] = torch.zeros((self.scaffold.graph.number_of_nodes(),)).float()
            spiking_neurons = [self.scaffold.graph.nodes[neuron]['neuron_number'] for neuron in input_values[t]]
            injection_tensors[t][spiking_neurons] = 1

        return injection_tensors

    def compile(self, scaffold, compile_args={}):
        # creates neuron populations and synapses
        self.scaffold = scaffold
        record = compile_args['record'] if 'record' in compile_args else 'output'

        if record == 'output':
            for node in self.scaffold.circuit.nodes:
                if 'layer' in self.scaffold.circuit.nodes[node]:
                    if self.scaffold.circuit.nodes[node]['layer'] == 'output':
                        for o_list in self.scaffold.circuit.nodes[node]['output_lists']:
                            for neuron in o_list:
                                self.scaffold.graph.nodes[neuron]['record'] = ['spikes']
        self.ds_graph = nx.convert_node_labels_to_integers(self.scaffold.graph, label_attribute='name')
        self.ds_graph.graph['has_delay'] = True
        if record == 'all':
            for neuron in self.ds_graph.nodes:
                self.ds_graph.nodes[neuron]['record'] = ['spikes']
        for neuron in self.ds_graph.nodes:
            if 'potential' not in self.ds_graph.nodes[neuron]:
                self.ds_graph.nodes[neuron]['potential'] = 0.0
            self.neuron_to_id_map[self.ds_graph.nodes[neuron]['name']] = neuron

        # Set initial inputs
        initial_spikes = self._get_initial_spike_times(scaffold.circuit)

        self.injection_values = self._create_ds_injection(initial_spikes)

    def run(self, n_steps=None, return_potentials=False):
        # runs circuit for n_steps then returns data
        # if not None raise error
        results = run_simulation(self.ds_graph, n_steps, self.injection_values)
        spike_result = pd.DataFrame({'time': [], 'neuron_number': []})

        if return_potentials:
            final_potentials = pd.DataFrame({'potential': [], 'neuron_number': []})

        for group in results:
            if 'spike_history' in results[group]:
                spike_history = results[group]['spike_history']
                for entry in spike_history:
                    mini_df = pd.DataFrame()
                    neurons = entry[1].tolist()
                    times = [entry[0]]*len(neurons)
                    mini_df['time'] = times
                    mini_df['neuron_number'] = neurons
                    spike_result = spike_result.append(mini_df, sort=True)
                if return_potentials and 'potential' in results[group]:
                    for i, potential in enumerate(results[group]['potential']):
                        final_potentials = final_potentials.append(
                                                                    {
                                                                      'potential': potential.tolist(),
                                                                      'neuron_number': i,
                                                                      },
                                                                    ignore_index=True,
                                                                    )

        if return_potentials:
            return (spike_result, final_potentials)
        else:
            return spike_result

    def cleanup(self):
        # Deletes/frees neurons and synapses
        self.neuron_to_id_map = {}
        self.scaffold = None
        self.ds_graph = None

    def reset(self):
        # resets time-step to 0 and resets neuron potentials
        for node in self.scaffold.graph.nodes:
            neuron_id = self.neuron_to_id_map[node]
            properties = self.scaffold.graph.nodes[node]
            if 'potential' in properties:
                self.ds_graph.nodes[neuron_id]['potential'] = properties['potential']
            else:
                self.ds_graph.nodes[neuron_id]['potential'] = 0.0

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
                brick_id = self.scaffold.brick_to_number[brick]
                changes = self.scaffold.circuit.nodes[brick_id]['brick'].set_parameters(parameters[brick])
                if changes:
                    neuron_props, synapse_props = changes
                    for neuron in neuron_props:
                        neuron_id = self.neuron_to_id_map[neuron]
                        properties = neuron_props[neuron]
                        for prop in properties:
                            self.ds_graph.nodes[neuron_id][prop] = properties[prop]

                    for synapse in synapse_props:
                        if type(synapse) is tuple:
                            pre, post = [self.neuron_to_id_map[p] for p in synapse]
                            properties = synapse_props[synapse]
                            for prop in properties:
                                self.ds_graph.edges[pre, post][prop] = properties[prop]
                        else:
                            neuron_id = self.neuron_to_id_map[synapse]
                            properties = synapse_props[synapse]
                            for edge in self.ds_graph.edges:
                                if edge[0] == neuron_id:
                                    for prop in properties:
                                        self.ds_graph.edges[edge][prop] = properties[prop]

        # Get new initial spike times
        initial_spikes = self._get_initial_spike_times(self.scaffold.circuit)

        self.injection_values = self._create_ds_injection(initial_spikes)
