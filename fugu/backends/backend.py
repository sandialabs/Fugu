#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 12:09:35 2019

@author: smusuva
"""

from abc import abstractmethod
from collections import deque
from warnings import warn

from ..utils.export_utils import results_df_from_dict

import pandas as pd
import networkx as nx

import abc
import sys
if sys.version_info >= (3, 4):
    ABC = abc.ABC
else:
    ABC = abc.ABCMeta('ABC', (), {'__slots__': ()})


class Backend(ABC):
    """
    Abstract Base Class definition of a Backend
    """
    def __init__(self):
        self.features = {
                          'supports_stepping': False,
                          'supports_streaming_input': False,
                          'supports_additive_leak': False,
                          'supports_hebbian_learning': False,
                          }

    def serve(
          self,
          scaffold,
          n_steps=None,
          max_steps=None,
          record='output',
          record_all=False,
          summary_steps=None,
          batch=True,
          backend_args={},
          ):
        if max_steps is not None:
            raise ValueError("Self-Halt is not yet implemented")
        if max_steps is not None and n_steps is not None:
            raise ValueError("Cannot specify both n_steps and max_steps")
        if record_all:
            record = 'all'

        self.embed(scaffold, record, backend_args)

        input_nodes = [node for node in scaffold.circuit.nodes if ('layer' in scaffold.circuit.nodes[node])
                       and (scaffold.circuit.nodes[node]['layer'] == 'input')]
        if batch:
            multi_runs = []
            single_runs = []
            for input_node in input_nodes:
                brick = scaffold.circuit.nodes[input_node]['brick']
                if brick.is_multi_runs:
                    multi_runs.append(brick)
                else:
                    single_runs.append(brick)

            if len(multi_runs) > 0:
                # Iterate through all multi_run bricks and build input values for each run.
                # If a brick does not explicitly specify an input for a given run then it is assumed that it uses
                #   its previous input.
                results = []

                static_values = {}
                for timestep in range(0, n_steps):
                    static_values[timestep] = deque()
                for node in single_runs:
                    for timestep, spike_list in enumerate(node):
                        static_values[timestep].extend(spike_list)

                max_runs = 0
                multi_inputs = {}
                for input_node in multi_runs:
                    multi_inputs[input_node] = list(iter(input_node))  # This feels pretty hacky
                    num_runs = len(multi_inputs[input_node])
                    if num_runs > max_runs:
                        max_runs = num_runs

                for i in range(max_runs):
                    # iterate through runs
                    input_values = {}
                    for timestep in range(0, n_steps):
                        input_values[timestep] = deque()
                    for key in static_values:
                        input_values[key].extend(static_values[key])
                    for node in multi_inputs:
                        if i < len(multi_inputs[node]):
                            input_values[0].extend(multi_inputs[node][i])
                        else:
                            number_finished += 1
                            input_values[0].extend(multi_inputs[node][-1])
                    results.append(self.batch(n_steps, input_values, backend_args))
            else:
                input_values = {}
                for timestep in range(0, n_steps):
                    input_values[timestep] = deque()
                for input_node in input_nodes:
                    for timestep, spike_list in enumerate(scaffold.circuit.nodes[input_node]['brick']):
                        input_values[timestep].extend(spike_list)

                results = self.batch(n_steps, input_values, backend_args)
        else:
            if not self.features['supports_stepping']:
                raise ValueError("Backend does not support stepping. Use a batch mode.")
            results = pd.DataFrame()
            halt = False
            step_number = 0
            while step_number < n_steps and not halt:
                input_values = deque()
                for input_node in input_nodes:
                    input_values.extend(next(scaffold.circuit.nodes[input_node]['brick'], []))
                result, halt = self.stream(scaffold, input_values, record, backend_args)
                results = results.append(result)

        self.cleanup()

        return results

    @abstractmethod
    def embed(self, scaffold, record, embedding_args={}):
        pass

    @abstractmethod
    def cleanup(self):
        pass

    @abstractmethod
    def stream(self, scaffold, input_values, stepping, record, backend_args):
        pass

    @abstractmethod
    def batch(self, n_steps, input_values=None, backend_args=None):
        pass


class snn_Backend(Backend):
    """
    Backend for Srideep's Noteworthy Network (SNN)
    """
    def __init__(self):
        super(Backend, self).__init__()
        self.results = []

    def _serve_fugu_to_snn(self, input_spike_lists, n_steps=1):
        for node, vals in self.fugu_circuit.nodes.data():
            if 'layer' in vals:
                if vals['layer'] == 'input':
                    input_values = {}
                    for neuron in self.fugu_circuit.nodes[node]['output_lists'][0]:
                        input_values[neuron] = []
                    for timestep in input_spike_lists:
                        spike_list = input_spike_lists[timestep]
                        for neuron in spike_list:
                            if neuron in input_values:
                                input_values[neuron].append(1)
                        for neuron in input_values:
                            if(len(input_values[neuron]) < timestep+1):
                                input_values[neuron].append(0)

                    for neuron in self.fugu_circuit.nodes[node]['output_lists'][0]:
                        self.nn.update_input_neuron(neuron, input_values[neuron])

        ''' Run the Simulator '''
        df = self.nn.run(n_steps=n_steps, debug_mode=False)
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
                    if df.loc[r][c] == 1:
                        col_list.append(c)
                    res[r] = col_list

            return res
        else:
            return df

    def embed(self, scaffold, record, embedding_args={}):
        '''
        Reads in a built fugu graph and converts it to a spiking neural network and runs it for n steps
        '''
        self.fugu_circuit = scaffold.circuit
        self.fugu_graph = scaffold.graph
        self.record = record
        if 'ds_format' in embedding_args:
            self.ds_format = embedding_args['ds_format']
        else:
            self.ds_format = False

        self._embed()

    def _embed(self):
        import fugu.backends.SpikingNeuralNetwork as snn
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
                        rc = True if self.record else vals.get('record', False)
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
                rc = True if self.record else params.get('record', False)
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

    def cleanup(self):
        pass

    def stream(self, scaffold, input_values, stepping, record, backend_args):
        warn("Stepping not supported yet.  Use a batching mode.")
        return None

    def batch(self, n_steps, input_values=None, backend_args=None):
        self._embed()

        results = self._serve_fugu_to_snn(input_values, n_steps=n_steps)
        return results_df_from_dict(results, 'time', 'neuron_number')


class ds_Backend(Backend):
    """
    Backend for the ds simulator
    """
    def __init__(self):
        super(Backend, self).__init__()
        self.results = []

    def _create_ds_injection(self, input_values):
        # find input nodes
        import torch

        injection_tensors = {}
        for t in range(len(input_values)):
            injection_tensors[t] = torch.zeros((self.scaffold.graph.number_of_nodes(),)).float()
            spiking_neurons = [self.scaffold.graph.nodes[neuron]['neuron_number'] for neuron in input_values[t]]
            injection_tensors[t][spiking_neurons] = 1

        return injection_tensors

    def embed(self, scaffold, record, embedding_args={}):
        self.scaffold = scaffold
        if record == 'output':
            for node in self.scaffold.circuit.nodes:
                if 'layer' in self.scaffold.circuit.nodes[node]:
                    if self.scaffold.circuit.nodes[node]['layer'] == 'output':
                        for o_list in self.scaffold.circuit.nodes[node]['output_lists']:
                            for neuron in o_list:
                                self.scaffold.graph.nodes[neuron]['record'] = ['spikes']
        self.ds_graph = nx.convert_node_labels_to_integers(self.scaffold.graph)
        self.ds_graph.graph['has_delay'] = True
        if record == 'all':
            for neuron in self.ds_graph.nodes:
                self.ds_graph.nodes[neuron]['record'] = ['spikes']
        for neuron in self.ds_graph.nodes:
            if 'potential' not in self.ds_graph.nodes[neuron]:
                self.ds_graph.nodes[neuron]['potential'] = 0.0

    def cleanup(self):
        pass

    def stream(self, scaffold, input_values, stepping, record, backend_args):
        warn("Stepping not supported yet.  Use a batching mode.")
        return None

    def batch(self, n_steps, input_values=None, backend_args=None):
        return_potentials = False
        if 'return_potentials' in backend_args:
            return_potentials = backend_args['return_potentials']

        injection_values = self._create_ds_injection(input_values)

        from fugu.backends.ds import run_simulation
        results = run_simulation(self.ds_graph, n_steps, injection_values)
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
                                                                      'potential':potential.tolist(),
                                                                      'neuron_number':i,
                                                                      },
                                                                    ignore_index=True,
                                                                    )

        if return_potentials:
            return (spike_result, final_potentials)
        else:
            return spike_result
