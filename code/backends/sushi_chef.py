#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 14:20:09 2019

@author: smusuva
"""

import Fugu.code.backends.SpikingNeuralNetwork as snn
import numpy as np

def serve_fugu_to_snn(fugu_circuit, fugu_graph, n_steps=1, record_all=False, ds_format=False):
    '''Reads in a built fugu graph and converts it to a spiking neural network 
    and runs it for n steps'''
 
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
                input_values = fugu_circuit.nodes[node]['brick'].get_input_value()    
                for neuron in fugu_circuit.nodes[node]['output_lists'][0]:
                    rc = True if record_all else vals.get('record', False)
                    neuron_dict[neuron] = snn.InputNeuron(neuron, record=rc)
                    idx = list(fugu_circuit.nodes[node]['output_lists'][0]).index(neuron)
                    neuron_dict[neuron].connect_to_input(input_values[idx])
                    nn.add_neuron(neuron_dict[neuron])
            if vals['layer'] == 'output':
                for olist in fugu_circuit.nodes[node]['output_lists']:
                    for neuron in olist:
                        params = fugu_graph.nodes[neuron]
                        th = params.get('threshold', 0.0)
                        rv = params.get('reset_voltage', 0.0)
                        lk = params.get('leakage_constant', 1.0)
                        vol =params.get('voltage', 0.0)
                        if 'potential' in params:
                            vol = params['potential']
                        if 'decay' in params:
                            lk = 1.0 - params['decay']
                        neuron_dict[neuron] = snn.LIFNeuron(neuron, threshold=th, reset_voltage=rv, leakage_constant=lk, voltage=vol, record=True)
                        nn.add_neuron(neuron_dict[neuron])
    #add other neurons from fugu_graph to spiking neural network
    #parse through the fugu_graph and if a neuron is not present in spiking neural network, add to it.                    
    for neuron, params in fugu_graph.nodes.data():
        if neuron not in neuron_dict.keys():
            th = params.get('threshold', 0.0)
            rv = params.get('reset_voltage', 0.0)
            lk = params.get('leakage_constant', 1.0)
            vol = params.get('voltage', 0.0)
            if 'potential' in params:
                vol = params['potential']
            if 'decay' in params:
                lk = 1.0 - params['decay']
            rc = True if record_all else params.get('record', False)
            neuron_dict[neuron] = snn.LIFNeuron(neuron, threshold=th, reset_voltage=rv, leakage_constant=lk, voltage=vol, record=rc)
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
    
    
    
    