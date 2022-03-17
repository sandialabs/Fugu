#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import deque
import numpy as np
from ..neuron.neuron import Neuron


class Synapse:
    """
    Synapses connect neurons in a neural network. The synapse class in a 
    spiking objects models a simple synapse type that scales the input by a 
    weight (double) and relays the information with a delay (non-negative int) of n 
    time-steps.   
    """
    
    def __init__(self, pre_neuron, post_neuron, delay=0, weight=1.0):
        """
        
        Parameters
        ----------
        pre_neuron : Neuron
           Neuron that provides input to the synapse
        post_neuron : Neuron
            Neuron that receives the signals from the synapse
        delay : non-negative Int, optional
            Number of time steps needed to relay the scaled spike. The default is 0.
        weight : double, optional
            Scaling value for incoming spike. The default is 1.0.

        Raises
        ------
        TypeError
            if pre and post neurons are not of type neurons.
        
        TypeError
            if delay is not of type Int.

        Returns
        -------
        None.

        """
        
        
        if not isinstance(pre_neuron, Neuron) or not isinstance(
                post_neuron, Neuron):
            raise TypeError(
                'Pre and Post Synanptic neurons must be of type Neuron')

        if type(delay) != int:
            raise TypeError(
                'delay must be an int - encode number of time steps')

        self._d = delay
        self._w = weight
        self._pre = pre_neuron
        self._post = post_neuron
        self._hist = deque(np.zeros(self._d))
        self.name = 's_' + self._pre.name + '_' + self._post.name

    def get_key(self):
        """
        
        Returns
        -------
        Tuple
            Pre and post neuron of the synapse.

        """
        return (self._pre, self._post)

    @property
    def pre_neuron(self):
        """
        Getter for pre_neuron.
        
        Returns
        -------
        Neuron
            Pre-synaptic neuron

        """
        return self._pre

    @property
    def post_neuron(self):
        """
        Getter for post_neuron.

        Returns
        -------
        Neuron
            post-synaptic neuron

        """
        return self._post

    @property
    def weight(self):
        """
        Getter for synaptic weight

        Returns
        -------
        Double
            scaling weight of the synapse.

        """
        return self._w

    @weight.setter
    def weight(self, new_weight=1.0):
        """
        Setter for synaptic weight

        Parameters
        ----------
        new_weight : Double, optional
            Sets the synaptic weight to a new value. The default is 1.0.

        Returns
        -------
        None.

        """
        self._w = new_weight

    @property
    def delay(self):
        """
        Getter for synaptic delay (in time steps)

        Returns
        -------
        Int
            Delay in time steps.

        """
        return self._d

    @delay.setter
    def delay(self, new_delay=0.0):
        """
        Setter for synaptic delay.

        Parameters
        ----------
        new_delay : Int, optional
            Sets the synaptic delay to a new value. The default is 0.0.

        Returns
        -------
        None.

        """
        self._d = new_delay

    def set_params(self, new_delay=0.0, new_weight=1.0):
        """
        Sets delay and weight of a synapse

        Parameters
        ----------
        new_delay : Int, optional
            Set delay to new value. The default is 0.0.
        new_weight : Double, optional
            Set weight to new value. The default is 1.0.

        Returns
        -------
        None.

        """
        self.weight = new_weight
        self.delay = new_delay

    def show_params(self):
        """
        Display the information of the synapse (pre-synaptic neuron, post-synaptic neuron,
        delay, and weight).

        Returns
        -------
        None.

        """
        print("Synapse {0} -> {1}:\n delay  : {2}\n weight : {3}".format(
            self._pre,
            self._post,
            self._d,
            self._w,
        ))

    def __str__(self):
        return "Synapse {0}({1}, {2})".format(self.name, self._d, self._w)

    def __repr__(self):
        return 's_' + self._pre.name + '_' + self._post.name

    def update_state(self):
        """
        updates the time evolution of the states for one time step. The spike information is 
        sent through a queue of length given by the delay and scaled by the weight value.

        Returns
        -------
        None.

        """
        
        if self._pre.spike:
            self._hist.append(self._w)
        else:
            self._hist.append(0.0)

        self._hist.popleft()
