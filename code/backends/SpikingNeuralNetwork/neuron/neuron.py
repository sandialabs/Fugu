#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 10:13:24 2018

@author: smusuva
"""
import abc

class Neuron(abc.ABC):
    @abc.abstractmethod
    def __init__(self, name=None, spike=False):
        self.name = name
        self.spike=False
        self.spike_hist = []
        
    @abc.abstractmethod
    def update_state(self):
        '''Update the time evolution of the neuron state'''


class LIFNeuron(Neuron):
    def __init__(self, name=None, threshold=0.0, reset_voltage=0.0, leakage_constant=1.0, 
                 voltage=0.0, record=False):
        super().__init__()
        self.name = name
        self._T = threshold
        self._R = reset_voltage
        self._m = leakage_constant
        self.v = voltage
        self.presyn = set()
        self.record = record
        
    def update_state(self):
        '''Update the states for one time step'''
        input_v = 0.0
        if self.presyn:
            for s in self.presyn:
#                print(s)
                input_v += s._hist[0]
        
#        self.v = (self._m * self.v) + input_v
        self.v = self.v + input_v
        
        if self.v > self._T:
            self.spike = True
            self.v = self._R
        else:
            self.spike = False
            self.v = self._m * self.v
            
        self.spike_hist.append(self.spike)
#
    def show_state(self):
        print("Neuron {0}: {1} volts, spike = {2}".format(self.name, self.v, self.spike))
        
    def show_params(self):
        print("Neuron '{0}':\n Threshold\t  :{1:2} volts,\n Reset voltage\t  :{2:1} volts,\n Leakage Constant :{3}\n".format(self.name,
                      self._T, self._R, self._m))
    
    def show_presynapses(self):
        if len(self.presyn) == 0:
            print("Neuron {0} receives no external input".format(self.name))
        elif len(self.presyn) == 1:
            print("{0} receives input via synapse: {1}".format(self.__repr__(), self.presyn))
        else:
            print("{0} receives input via synapses: {1}".format(self.__repr__(), self.presyn))
    
    @property
    def threshold(self):
        return self._T
    
    @threshold.setter
    def threshold(self, new_threshold):
        self._T = new_threshold
    
    @property
    def reset_voltage(self):
        return self._R
    
    @reset_voltage.setter
    def reset_voltage(self, new_reset_v):
        self._R = new_reset_v
    
    @property
    def leakage_constant(self):
        return self._m
    
    @leakage_constant.setter
    def leakage_constant(self, new_leak_const):
        self._m = new_leak_const
    
    @property
    def voltage(self):
        return self.v
    
    def __str__(self):
        return "LIFNeuron {0}({1}, {2}, {3})".format(self.name, self._T, self._R, self._m)
    
    def __repr__(self):
        return "LIFNeuron {0}".format(self.name)
    
class InputNeuron(Neuron):
    '''Input Neuron. Inherits from class Neuron'''
    
    def __init__(self, name=None, threshold=0.1, voltage=0.0, record=False):
        super().__init__()
        self.name = name
        self._T = threshold
        self.v = voltage
        #self.spike = spike
        self._it = None
        self.record = record
        
#    def _create_iterator(self, input_iterable):
#        '''Create an iterable from the input'''
#            self._it = iter(input_iterable)
        
    def connect_to_input(self, in_stream):
        if not hasattr(in_stream, '__iter__'):
            raise TypeError(f'{in_stream} must be iterable')
        else:
            self._it = iter(in_stream)
        
    def update_state(self):
        try:
            n = next(self._it)
#            if not isinstance(n, int) and not isinstance(n, float):
#                raise TypeError('Inputs must be int or float')
            
            self.v = n
            
            if self.v > self._T:
                self.spike = True
                self.v = 0
            else:
                self.spike = False
                self.v = 0
        except StopIteration:
            self.spike = False
            self.v = 0
       
    @property
    def threshold(self):
        return self._T
    
    @threshold.setter
    def threshold(self, new_threshold):
        self._T = new_threshold
    
    @property
    def voltage(self):
        return self.v
    
        
    def __str__(self):
        return f'InputNeuron {self.name}'
    
    def __repr__(self):
        return f'InputNeuron {self.name}'