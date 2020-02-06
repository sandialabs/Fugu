#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from .backend import Backend

from ..utils.misc import CalculateSpikeTimes

import nxsdk.api.n2a as nx      # Nx API



import matplotlib.pyplot as plt
import matplotlib as mpl
haveDisplay = "DISPLAY" in os.environ
if not haveDisplay:
    mpl.use('Agg')

from warnings import warn

class LoihiBackend(Backend):
    def __init__(self):
        super(Backend, self).__init__()
        
    def _build_network(self):
        net = nx.NxNet()
        fugu_loihi_map = {}
        loihi_outputs = {}
        probeParameters = [nx.ProbeParameter.SPIKE]
        
        ip_spikes = CalculateSpikeTimes(self.fugu_circuit, main_key='neuron_name')
        
        max_wt = 0.0
        min_wt = 0.0
        wt_bdry = 0.0
        
        for n1, n2, props in self.fugu_graph.edges.data():
            wt = props.get('weight', 1.0)
            if wt > max_wt:
                max_wt = wt
            if wt < min_wt:
                min_wt = wt
        
        if abs(min_wt) >= abs(max_wt):
            wt_bdry = abs(min_wt)
        else:
            wt_bdry = abs(max_wt)
        
        
        for node, vals in self.fugu_circuit.nodes.data():
            if 'layer' in vals:
                if vals['layer'] == 'input':
                    for neuron in self.fugu_circuit.nodes[node]['output_lists'][0]:
                        fugu_loihi_map[neuron] = net.createSpikeGenProcess(2)
                        fugu_loihi_map[neuron].addSpikes([0], ip_spikes[neuron])
                if vals['layer'] == 'output':
                    for olist in self.fugu_circuit.nodes[node]['output_lists']:
                        for neuron in olist:
                            props = self.fugu_graph.nodes[neuron]
                            th = props.get('threshold', 0.0)
                            decay = props.get('decay', 0.0)
                            vol = props.get('voltage', 0.0)
                            prob = props.get('p', 1.0)
                            if prob < 1.0:
                                warn('Probabilistic neurons not Implemented. Treating neurons as deterministic neurons')
                            if 'potential' in props:
                                vol = props['potential']
                            if 'leakage_constant' in props:
                                decay = 1.0 - props['leakage_constant']
                            
                            th = self._cvt_weight(th, floor=-wt_bdry, ceiling = wt_bdry)
                            decay = self._cvt_weight(decay, floor=-wt_bdry, ceiling = wt_bdry)
                            vol = self._cvt_weight(vol, floor=-wt_bdry, ceiling = wt_bdry)
                            
                            prototype = nx.CompartmentPrototype(vThMant                  = th,
                                                                compartmentVoltage       = vol,
                                                                compartmentVoltageDecay  = decay)
                            fugu_loihi_map[neuron] = self.net.createCompartment(prototype)
                            loihi_outputs[neuron] = fugu_loihi_map[neuron].probe(probeParameters)
                            
        for neuron, props in self.fugu_graph.nodes.data():
            if neuron not in fugu_loihi_map.keys():
                th = props.get('threshold', 0.0)
                decay = props.get('decay', 0.0)
                vol = props.get('voltage', 0.0)
                prob = props.get('p', 1.0)
                if prob < 1.0:
                    warn('Probabilistic neurons not Implemented. Treating neurons as deterministic neurons')
                if 'potential' in props:
                    vol = props['potential']
                if 'leakage_constant' in props:
                    decay = 1.0 - props['leakage_constant']
                    
                th = self._cvt_weight(th, floor=-wt_bdry, ceiling = wt_bdry)
                decay = self._cvt_weight(decay, floor=-wt_bdry, ceiling = wt_bdry)
                vol = self._cvt_weight(vol, floor=-wt_bdry, ceiling = wt_bdry)
                
                prototype = nx.CompartmentPrototype(vThMant                  = th,
                                                    compartmentVoltage       = vol,
                                                    compartmentVoltageDecay  = decay)
                fugu_loihi_map[neuron] = self.net.createCompartment(prototype)            
                
        
        '''Add connections'''
        
        for n1, n2, props in self.fugu_graph.edges.data():
            delay = int(props.get('delay', 1))
            wt = props.get('weight', 1.0)
            wt = self._cvt_weight(wt, floor = -wt_bdry, ceiling = wt_bdry)
            if delay <= 6:
                conn_prototype = nx.ConnectionPrototype(weight          = wt,
                                                        numWeightBits   = 8,
                                                        compressionMode = 3,
                                                        delay           = delay)
                fugu_loihi_map[n1].connect(fugu_loihi_map[n2], prototype=conn_prototype)
            else:
                quot = delay//6
                rem = delay%6
                for _ in range(quot-1):
                    conn_prototype = nx.ConnectionPrototype(weight          = wt,
                                                            numWeightBits   = 8,
                                                            compressionMode = 3,
                                                            delay           = 6)
                    temp_neuron_proto = nx.CompartmentPrototype(vThMant     = 100)
                    neuron_num = len(fugu_loihi_map)
                    fugu_loihi_map[neuron_num] = self.net.createCompartment(temp_neuron_proto)
                    fugu_loihi_map[n1].connect(fugu_loihi_map[neuron_num], prototype = conn_prototype)
                if rem:
                    conn_prototype = nx.ConnectionPrototype(weight          = wt,
                                                            numWeightBits   = 8,
                                                            compressionMode = 3,
                                                            delay           = rem)
                    temp_neuron_proto = nx.CompartmentPrototype(vThMant     = 100)
                    neuron_num = len(fugu_loihi_map)
                    fugu_loihi_map[neuron_num] = self.net.createCompartment(temp_neuron_proto)
                    fugu_loihi_map[n1].connect(fugu_loihi_map[neuron_num], prototype = conn_prototype)
           
            

                                                                             
    def _cvt_weight(self, weight, floor=-2.0, ceiling=2.0, full_precision=True, clip=True):
        """ Maps a floating point weight from the range (floor, ceiling) to the range (-254, 254) """
        if clip:
            percentage = (min(max(weight, floor), ceiling) - floor) / (ceiling - floor)
        else:
            percentage = (weight - floor) / (ceiling - floor)
        if full_precision:
            loihi_weight = int(round((-255 + (255*2.0 * percentage))))
            loihi_weight = max(min(255, loihi_weight), -255) if clip else loihi_weight
        else:
            loihi_weight = int(round((-127 + (254 * percentage))*2.0))
            loihi_weight = loihi_weight if loihi_weight % 2 == 0 else (loihi_weight + (1 if loihi_weight > 0 else -1))
            loihi_weight = max(min(254, loihi_weight), -254) if clip else loihi_weight
        return loihi_weight

    def compile(self, scaffold, compile_args={}):
        self.fugu_circuit = scaffold.circuit
        self.fugu_graph = scaffold.graph
        self.brick_to_number = scaffold.brick_to_number
        
        self._build_network()
        
    def run(self, n_steps = 10):
        print('running simulation...')
        self.net.run(n_steps)
        self.net.disconnect()
        for k, v in self.loihi_outputs:
            print(f'probe[{k}] =', self.loihi_output[v].data[:])
            
    def cleanup(self):
        # Deletes/frees neurons and synapses
        pass

    def reset(self):
        # resets time-step to 0 and resets neuron/synapse properties
        self._build_network()
