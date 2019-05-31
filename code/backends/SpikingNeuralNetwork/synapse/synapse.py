#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 10:16:18 2018

@author: smusuva
"""

from collections import deque
import numpy as np
from ..neuron.neuron import Neuron

class Synapse:
    def __init__(self, pre_neuron, post_neuron, delay=0, weight=1.0):
        if not isinstance(pre_neuron, Neuron) or not isinstance(post_neuron, Neuron):
            raise TypeError('Pre and Post Synanptic neurons must be of type Neuron')
        
        if type(delay) != int:
            raise TypeError('delay must be an int - encode number of time steps')
        
        self._d = delay
        self._w = weight
        self._pre = pre_neuron
        self._post = post_neuron
        self._hist = deque(np.zeros(self._d))
        self.name = 's_' + self._pre.name + '_' + self._post.name
    
    @property    
    def pre_neuron(self):
        return self._pre
    
    @property
    def post_neuron(self):
        return self._post
    
    @property
    def weight(self):
        return self._w
    
    @weight.setter
    def weight(self, new_weight=1.0):
        self._w = new_weight
    
    @property
    def delay(self):
        return self._d
    
    @delay.setter
    def delay(self, new_delay=0.0):
        self._d = new_delay
    
    
    def set_params(self, new_delay=0.0, new_weight=1.0):
        self.set_weight(new_weight)
        self.set_delay(new_delay)
    
    def show_params(self):
        print("Synapse {0} -> {1}:\n delay  : {2}\n weight : {3}".format(self._pre, 
              self._post, self._d, self._w ))
        
    def __str__(self):
        return "Synapse {0}({1}, {2})".format(self.name, self._d, self._w)
    
    def __repr__(self):
        return 's_' + self._pre.name + '_' + self._post.name
    
    def update_state(self):
        '''update the time evolution of states for one time step'''
        if self._pre.spike: 
            self._hist.append(self._w)
        else:
            self._hist.append(0.0)
        
        self._hist.popleft()
       