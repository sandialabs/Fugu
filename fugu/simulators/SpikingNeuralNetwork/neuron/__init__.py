#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 09:18:44 2018

@author: smusuva
"""

from .neuron import Neuron, LIFNeuron, InputNeuron


#from collections import deque
#from collections import Iterable
#import numpy as np
#import pandas as pd
#
#class Neuron:
#    def __init__(self, name=None, threshold=0.0, reset_voltage=0.0, leakage_constant=1.0, 
#                 voltage=0.0, spike=False):
#        self._name = name
#        self._T = threshold
#        self._R = reset_voltage
#        self._m = leakage_constant
#        self._v = voltage
#        self._spike = spike
#        self._presyn = set()
#        
#    def update_state(self):
#        input_v = 0.0
#        if self._presyn:
#            for s in self._presyn:
#                input_v += s._hist[0]
#        
#        #if self._v > self._R:
#        self._v = (self._m * self._v) + input_v
#        #else:
#        #    self._v = self._R + input_v
#        
#        if self._v >= self._T:
#            self._spike = True
#            self._v = self._R
#        else:
#            self._spike = False
#        
#    def show_state(self):
#        print("Neuron {0}: {1} volts, spike = {2}".format(self._name, self._v, self._spike))
#        
#    def show_params(self):
#        print("Neuron '{0}':\n Threshold\t  :{1:2} volts,\n Reset voltage\t  :{2:1} volts,\n Leakage Constant :{3}\n".format(self._name,
#                      self._T, self._R, self._m))
#    
#    def show_presynapses(self):
#        if len(self._presyn) == 0:
#            print("Neuron {0} receives no external input".format(self._name))
#        elif len(self._presyn) == 1:
#            print("{0} receives input via synapse: {1}".format(self.__repr__(), self._presyn))
#        else:
#            print("{0} receives input via synapses: {1}".format(self.__repr__(), self._presyn))
#        
#    def set_threshold(self, new_threshold):
#        self._T = new_threshold
#        
#    def set_reset_voltage(self, new_reset_v):
#        self._R = new_reset_v
#        
#    def set_leakage_constant(self, new_leak_const):
#        self._m = new_leak_const
#
#    def __str__(self):
#        return "Neuron {0}({1}, {2}, {3})".format(self._name, self._T, self._R, self._m)
#    
#    def __repr__(self):
#        return "Neuron {0}".format(self._name)
#    
#    
#    
#class Synapse:
#    def __init__(self, pre_syn_neuron, post_syn_neuron, delay=0, weight=1.0):
#        if type(pre_syn_neuron) != Neuron or type(post_syn_neuron) != Neuron:
#            raise TypeError('Pre and Post Synanptic neurons must be of type Neuron')
#        
#        if type(delay) != int:
#            raise TypeError('delay must be an int - encode number of time steps')
#            
#        self._d = delay
#        self._w = weight
#        self._pre = pre_syn_neuron
#        self._post = post_syn_neuron
#        self._name = 's_' + str(self._pre) + '_' + str(self._post)
#        self._hist = deque(np.zeros(self._d))
#        
#    def get_pre(self):
#        return self._pre
#    
#    def get_post(self):
#        return self._post
#    
#    def set_weight(self, new_weight=1.0):
#        self._w = new_weight
#    
#    def set_delay(self, new_delay=0.0):
#        self._d = new_delay
#    
#    def set_params(self, new_delay=0.0, new_weight=1.0):
#        self.set_weight(new_weight)
#        self.set_delay(new_delay)
#    
#    def show_params(self):
#        print("Synapse {0} -> {1}:\n delay  : {2}\n weight : {3}".format(self._pre, 
#              self._post, self._d, self._w ))
#        
#    def __str__(self):
#        return "Synapse {0}({1}, {2})".format(self._name, self._d, self._w)
#    
#    def __repr__(self):
#        return 's_' + self._pre._name + '_' + self._post._name
#    
#    def update_state(self):
#        self._hist.popleft()
#        if self._pre._spike: 
#            self._hist.append(self._w)
#        else:
#            self._hist.append(0.0)
#            
#
#class NeuralNetwork:
#    def __init__(self):
#        self._nrns = set()
#        self._synps = set()
#        self._nrn_count = 0
#        #self._inmap = {}
#        
#    def add_neuron(self, new_neuron=None):
#        if not new_neuron:
#            self._nrn_count += 1
#            self._nrns.add(Neuron(str(self._nrn_count)))
#        elif type(new_neuron) == str:
#            self._nrn_count += 1
#            self._nrns.add(Neuron(new_neuron))
#        elif type(new_neuron) == Neuron:
#            self._nrn_count += 1
#            self._nrns.add(new_neuron)
#        else:
#            raise TypeError("{0} must be of type Neuron or str".format(new_neuron))
#            
#    def add_multiple_neurons(self, neuron_iterable=None):
#        if not neuron_iterable:
#            self.add_neuron()
#            
#        if not isinstance(neuron_iterable, Iterable):
#            raise TypeError("{0} is not iterable".format(neuron_iterable))
#            
#        for n in neuron_iterable:
#            self.add_neuron(n)
#            
#    def list_neurons(self):
#        print("Neurons: {", end='')
#        for n in self._nrns:
#            print("{},".format(n._name), end=" ")
#        print("\b\b}")
#            
#    def add_synapse(self, new_synapse=None):
#        if not new_synapse:
#            raise TypeError("Needs synapse object with pre and post neurons")
#        elif type(new_synapse) == tuple and len(new_synapse) >=2 and len(new_synapse)<5:
#            tmpsyn = Synapse(*new_synapse)
#        elif type(new_synapse) == Synapse:
#            tmpsyn = new_synapse
#        else:
#            raise TypeError("Must provide Synapse Object")
#        
##        if tmpsyn not in self._synps:
##            self._synps.add(tmpsyn)
##            self.update_network(tmpsyn)
##        else:
##            print("Warning! Not Added! {0} already defined in network. (Use <synapse>.set_params() to update synapse)".format(tmpsyn))
##            print()
#        self._synps.add(tmpsyn)
#        self.update_network(tmpsyn)
#        
#    def add_multiple_synapses(self, synapse_iterable=None):
#        if not isinstance(synapse_iterable, Iterable):
#            raise TypeError("{0} is not iterable".format(synapse_iterable))
#            
#        for s in synapse_iterable:
#            self.add_synapse(s)
#    
#    ### NOT SURE IF THIS IS NEEDED! ### 
#    #def build_network(self):
#    #    for s in self._synps:
#    #        if s._post in self._inmap.keys():
#    #            self._inmap[s._post].add(s)
#    #        else:
#    #            self._inmap[s._post] = {s}
#    ###
#    
#    # Will be called automatically if a synapse is added
#    def update_network(self, new_synapse):
#        #self._inmap[new_synapse._post] = new_synapse
#        new_synapse._post._presyn.add(new_synapse)
#        
#    def step(self):
#        for n in self._nrns:
#            n.update_state()
#     
#        for s in self._synps:
#            s.update_state()
#            
#    def run(self, n_steps=1):
#        tempdct = {0:[]}
#        nrn_list = []
#        for n in self._nrns:
#            nrn_list.append(n._name)
#            if n._spike:
#                tempdct[0].append(1)
#            else:
#                tempdct[0].append(0)
#        for t in range(1, n_steps):
#            self.step()
#            tempdct[t] = []
#            for n in self._nrns:
#                if n._spike:
#                    tempdct[t].append(1)
#                else:
#                    tempdct[t].append(0)
#        
#                   
#                df = pd.DataFrame.from_dict(tempdct, orient='index', columns=nrn_list)
#                df.columns.rename('Neurons',inplace=True)
#                df.index.rename('Time',inplace=True)
#        return df
#                
#        
#
