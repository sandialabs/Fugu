#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 12:09:35 2019

@author: smusuva
"""
import abc
import sys
if sys.version_info >= (3, 4):
    ABC = abc.ABC
else:
    ABC = abc.ABCMeta('ABC', (), {'__slots__': ()})
from abc import abstractmethod
from collections import deque
import pandas as pd
from warnings import warn
from ..utils.export_utils import results_df_from_dict
import networkx as nx

class Backend(ABC):
    """Abstract Base Class definition of a Backend"""
    def __init__(self):
        self.features = {'supports_stepping':False,
                         'supports_streaming_input':False,
                         'supports_additive_leak':False,
                         'supports_hebbian_learning':False}
        
    def serve(self,
            scaffold,
            n_steps=None,
            max_steps=None,
            record='output',
            record_all=False,
            summary_steps=None,
            batch=True,
            backend_args={}):
        if max_steps is not None:
            raise ValueError("Self-Halt is not yet implemented")
        if max_steps is not None and n_steps is not None:
            raise ValueError("Cannot specify both n_steps and max_steps")
        if record_all:
            record = 'all'
           
        input_nodes = [node  for node in scaffold.circuit.nodes if ('layer' in scaffold.circuit.nodes[node] )
                       and (scaffold.circuit.nodes[node]['layer'] == 'input')]
        if batch:
            input_values = {}
            for timestep in range(0,n_steps):
                input_values[timestep]=deque()
            for input_node in input_nodes:
                for timestep, spike_list in enumerate(scaffold.circuit.nodes[input_node]['brick']):
                    input_values[timestep].extend(spike_list)
            results = self.batch(scaffold, input_values, n_steps, record, backend_args)
        else:
            if not self.features['supports_stepping']:
                raise ValueError("Backend does not support stepping. Use a batch mode.")
            results = pd.DataFrame()
            halt = False
            step_number = 0
            while step_number < n_steps and not halt:
                input_values = deque()
                for input_node in input_nodes:
                    input_values.extend(next(scaffold.circuit.nodes[input_node]['brick'],[]))
                result, halt = self.stream(scaffold, input_values,  record, backend_args)
                results = results.append(result)
        return results
    
    @abstractmethod
    def stream(self,
               scaffold,
               input_values,
               stepping,
               record,
               backend_args):
        pass
    
    @abstractmethod
    def batch(self,
             scaffold,
             input_values,
             n_steps,
             record,
             backend_args):
        pass

class snn_Backend(Backend):
    """Backend for Srideep's Noteworthy Network (SNN)"""
    def __init__(self):
        super(Backend, self).__init__()

    def _serve_fugu_to_snn(self, fugu_circuit, fugu_graph, n_steps=1, record_all=False, ds_format=False):
        '''Reads in a built fugu graph and converts it to a spiking neural network 
        and runs it for n steps'''
        import fugu.backends.SpikingNeuralNetwork as snn
        nn = snn.NeuralNetwork()
        neuron_dict = {}
        
        
        ''' Add Neurons '''
        #Add in input and output neurons. Use the fugu_circuit information to identify input and output layers
        #For input nodes, create input neurons and identity and associate the appropriate inputs to it
        #for output neurons, obtain neuron parameters from fugu_graph and create LIFNeurons
        #Add neurons to spiking neural network
        for node, vals in fugu_circuit.nodes.data():
            if 'layer' in vals: 
                if vals['layer'] == 'input':
                    
                    input_spike_lists = [input_spikes for input_spikes in fugu_circuit.nodes[node]['brick']]
                    input_values = {}
                    for neuron in fugu_circuit.nodes[node]['output_lists'][0]:
                        input_values[neuron] = []
                    for timestep, spike_list in enumerate(input_spike_lists):
                        for neuron in spike_list:
                            input_values[neuron].append(1)
                        for neuron in input_values:
                            if(len(input_values[neuron]) < timestep+1):
                                input_values[neuron].append(0)
                            
                    
                    for neuron in fugu_circuit.nodes[node]['output_lists'][0]:
                        rc = True if record_all else vals.get('record', False)
                        neuron_dict[neuron] = snn.InputNeuron(neuron, record=rc)
                        neuron_dict[neuron].connect_to_input(input_values[neuron])
                        nn.add_neuron(neuron_dict[neuron])
                if vals['layer'] == 'output':
                    for olist in fugu_circuit.nodes[node]['output_lists']:
                        for neuron in olist:
                            params = fugu_graph.nodes[neuron]
                            th = params.get('threshold', 0.0)
                            rv = params.get('reset_voltage', 0.0)
                            lk = params.get('leakage_constant', 1.0)
                            vol =params.get('voltage', 0.0)
                            prob = params.get('p', 1.0)
                            if 'potential' in params:
                                vol = params['potential']
                            if 'decay' in params:
                                lk = 1.0 - params['decay']
                            neuron_dict[neuron] = snn.LIFNeuron(neuron, threshold=th, reset_voltage=rv, leakage_constant=lk, voltage=vol, p=prob, record=True)
                            nn.add_neuron(neuron_dict[neuron])
        #add other neurons from fugu_graph to spiking neural network
        #parse through the fugu_graph and if a neuron is not present in spiking neural network, add to it.                    
        for neuron, params in fugu_graph.nodes.data():
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
                rc = True if record_all else params.get('record', False)
                neuron_dict[neuron] = snn.LIFNeuron(neuron, threshold=th, reset_voltage=rv, leakage_constant=lk, voltage=vol, p=prob, record=rc)
                nn.add_neuron(neuron_dict[neuron])
        
        ''' Add Synapses '''
        #add synapses from fugu_graph edge information        
        for n1, n2, params in fugu_graph.edges.data():
            delay = int(params.get('delay', 1))
            wt = params.get('weight', 1.0)
            syn = snn.Synapse(neuron_dict[n1], neuron_dict[n2], delay=delay, weight=wt)
            nn.add_synapse(syn)
        
        del neuron_dict
        
        ''' Run the Simulator '''
        df = nn.run(n_steps=n_steps, debug_mode=False)
        res = {}
        #if ds format is required, convert neuron names to numbers and return dictionary
        #else return dataframe
        if ds_format:
            numerical_cols = {}
            for c in df.columns:
                numerical_cols[c] = fugu_graph.nodes[c]['neuron_number']
            df = df.rename(index=int, columns=numerical_cols) 
            
            for r in df.index:
                l = []
                for c in df.columns:
                    if df.loc[r][c] == 1:
                        l.append(c)
                    res[r] = l
    
            return res
        else:
            return df
    
    def stream(self,
               scaffold,
               input_values,
               stepping,
               record,
               backend_args):
        warn("Stepping not supported yet.  Use a batching mode.")
        return None
    
    def batch(self,
             scaffold,
             input_values,
             n_steps,
             record,
             backend_args):
        spike_result = self._serve_fugu_to_snn(scaffold.circuit, scaffold.graph, n_steps=n_steps, record_all=True if record=='all' else False, ds_format=True)
        return results_df_from_dict(spike_result,'time','neuron_number')
    
class ds_Backend(Backend):
    """Backend for the ds simulator"""
    def __init__(self):
        super(Backend,self).__init__()
        
    def _create_ds_injection(self,scaffold,input_values):
        #find input nodes
        import torch

        injection_tensors = {}
        for t in range(len(input_values)):
            injection_tensors[t] = torch.zeros((scaffold.graph.number_of_nodes(),)).float()
            spiking_neurons = [scaffold.graph.nodes[neuron]['neuron_number'] for neuron in input_values[t]]
            injection_tensors[t][spiking_neurons] = 1
        return injection_tensors    
    
    def stream(self,
               scaffold,
               input_values,
               stepping,
               record,
               backend_args):
        warn("Stepping not supported yet.  Use a batching mode.")
        return None

    
    def batch(self,
              scaffold,
              input_values,
              n_steps,
              record,
              backend_args):
        from fugu.backends.ds import run_simulation
        injection_values = self._create_ds_injection(scaffold,input_values)
        if record =='output':
            for node in scaffold.circuit.nodes:
                if 'layer' in scaffold.circuit.nodes[node] and scaffold.circuit.nodes[node]['layer'] == 'output':
                    for o_list in scaffold.circuit.nodes[node]['output_lists']:
                        for neuron in o_list:
                            scaffold.graph.nodes[neuron]['record'] = ['spikes']
        ds_graph = nx.convert_node_labels_to_integers(scaffold.graph)
        ds_graph.graph['has_delay']=True
        if record == 'all':
            for neuron in ds_graph.nodes:
                ds_graph.nodes[neuron]['record'] = ['spikes']
        for neuron in ds_graph.nodes:
            if 'potential' not in ds_graph.nodes[neuron]:
                ds_graph.nodes[neuron]['potential'] = 0.0
        results = run_simulation(ds_graph,
                                 n_steps,
                                 injection_values)
        spike_result = pd.DataFrame({'time':[],'neuron_number':[]})
        for group in results:
            if 'spike_history' in results[group]:
                spike_history = results[group]['spike_history']
                for entry in spike_history:
                    mini_df = pd.DataFrame()
                    neurons = entry[1].tolist()
                    times = [entry[0]]*len(neurons)
                    mini_df['time'] = times
                    mini_df['neuron_number'] = neurons
                    spike_result = spike_result.append(mini_df)
        return spike_result
