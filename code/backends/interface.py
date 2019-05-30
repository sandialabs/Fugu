#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 14:20:09 2019

@author: smusuva
"""

import backends.SpikingNeuralNetwork as snn


def digest_fugu(fugu_circuit, fugu_graph, n_steps=1, record_all=False):
    '''Reads in a built fugu graph and converts it to a spiking neural network 
    and runs it for n steps'''
 
    nn = snn.NeuralNetwork()
    neuron_dict = {}
    
    for node, vals in fugu_circuit.nodes.data():
        if 'layer' in vals: 
            if vals['layer'] == 'input':
                input_values = fugu_circuit.nodes[node]['brick'].get_input_value()
                for neuron in fugu_circuit.nodes[node]['output_lists'][0]:
                    rc = True if record_all else vals.get('record', False)
                    neuron_dict[neuron] = snn.InputNeuron(neuron, record=rc)
                    idx = list(fugu_graph.nodes).index(neuron)
                    neuron_dict[neuron].connect_to_input(input_values[idx])
                    nn.add_neuron(neuron_dict[neuron])
            if vals['layer'] == 'output':
                for olist in fugu_circuit.nodes[node]['output_lists']:
                    for neuron in olist:
                        th = vals.get('threshold', 0.0)
                        rv = vals.get('reset_voltage', 0.0)
                        lk = vals.get('leakage_constant', 1.0)
                        vol = vals.get('voltage', 0.0)
                        if 'potential' in vals:
                            vol = vals['potential']
                        neuron_dict[neuron] = snn.LIFNeuron(neuron, threshold=th, reset_voltage=rv, leakage_constant=lk, voltage=vol, record=True)                            
                        nn.add_neuron(neuron_dict[neuron])
                        
    for neuron, params in fugu_graph.nodes.data():
        if neuron not in neuron_dict.keys():
            th = params.get('threshold', 0.0)
            rv = params.get('reset_voltage', 0.0)
            lk = params.get('leakage_constant', 1.0)
            vol = params.get('voltage', 0.0)
            if 'potential' in params:
                vol = params['potential']
            rc = True if record_all else params.get('record', False)
            neuron_dict[neuron] = snn.LIFNeuron(neuron, threshold=th, reset_voltage=rv, leakage_constant=lk, voltage=vol, record=rc) 
            nn.add_neuron(neuron_dict[neuron])
            
    for n1, n2, params in fugu_graph.edges.data():
        delay = int(params.get('delay', 1))
        wt = params.get('weight', 1.0)
        syn = snn.Synapse(neuron_dict[n1], neuron_dict[n2], delay=delay, weight=wt)
        nn.add_synapse(syn)
        
    del neuron_dict
    df = nn.run(n_steps=n_steps, debug_mode=False)
    return df
    
    
    
    