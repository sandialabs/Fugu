#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import deque

import numpy as np

# from ..neuron.neuron import Neuron
from fugu.simulators.SpikingNeuralNetwork import Neuron
from fugu.utils.types import float_types, int_types
from fugu.utils.validation import int_to_float, validate_type


class Synapse:
    """
    Synapses connect neurons in a neural network. The synapse class in a
    spiking objects models a simple synapse type that scales the input by a
    weight (double) and relays the information with a delay (non-negative int) of n
    time-steps.
    """

    def __init__(self, pre_neuron, post_neuron, delay=1, weight=1.0):
        """
        Parameters:
            pre_neuron (any): Neuron that provides input to the synapse
            post_neuron (any): Neuron that receives the signals from the synapse
            delay (int) : non-negative Int, optional.  Number of time steps needed to relay the scaled spike. The default is 1.
            weight (double): optional.  Scaling value for incoming spike. The default is 1.0.

        Raises:
            TypeError: if pre and post neurons are not of type neurons
            TypeError: if delay is not of type Int
            ValueError: if delay is less than 1

        Returns:
            none
        """

        if not isinstance(pre_neuron, Neuron) or not isinstance(post_neuron, Neuron):
            raise TypeError("Pre and Post Synanptic neurons must be of type Neuron")

        weight = int_to_float(weight)
        validate_type(delay, int_types)
        validate_type(weight, float_types)

        if delay < 1:
            raise ValueError("delay must be a strictly positive (>0) int value")

        self._d = delay
        self._w = weight
        self._pre = pre_neuron
        self._post = post_neuron
        self._hist = deque(np.zeros(self._d))
        self.name = "s_" + self._pre.name + "_" + self._post.name

    def get_key(self):
        """
        Returns:
            self._pre (tuple): Pre neuron of the synapse
            self._post (tuple) post neuron of synapse

        """
        return (self._pre, self._post)

    @property
    def pre_neuron(self):
        """
        Getter for pre_neuron.

        Returns:
            self._pre: Neuron Pre-synaptic neuron
        """
        return self._pre

    @property
    def post_neuron(self):
        """
        Getter for post_neuron.

        Returns:
            Neuron: post-synaptic neuron
        """
        return self._post

    @property
    def weight(self):
        """
        Getter for synaptic weight

        Returns:
            self._w (double): scaling weight of the synapse.
        """
        return self._w

    @weight.setter
    def weight(self, new_weight=1.0):
        """
        Setter for synaptic weight

        Parameters:
            new_weight (float): Double, optional.  Sets the synaptic weight to a new value. The default is 1.0.
        Returns:
            none
        """

        new_weight = int_to_float(new_weight)
        validate_type(new_weight, float_types)
        self._w = new_weight

    @property
    def delay(self):
        """
        Getter for synaptic delay (in time steps)

        Returns:
            self._d (int): delay in time steps
        """
        return self._d

    @delay.setter
    def delay(self, new_delay=1):
        """
        Setter for synaptic delay.

        Parameters:
            new_delay (int): optional Sets the synaptic delay to a new value. The default is 1.0.

        Returns:
            None
        """

        validate_type(new_delay, int_types)

        if new_delay < 1:
            raise ValueError("delay must be a strictly positive (>0) int value")

        self._d = new_delay

    def set_params(self, new_delay=1, new_weight=1.0):
        """
        Sets delay and weight of a synapse

        Parameters:
            new_delay (int): optional Set delay to new value. The default is 1.0.
            new_weight (Double): optional. Set weight to new value. The default is 1.0.

        Returns:
            None
        """

        new_weight = int_to_float(new_weight)
        validate_type(new_delay, int_types)
        validate_type(new_weight, float_types)

        self.delay = new_delay
        self.weight = new_weight

    def show_params(self):
        """
        Display the information of the synapse (pre-synaptic neuron, post-synaptic neuron,
        delay, and weight).

        Returns:
            None
        """

        print(
            "Synapse {0} -> {1}:\n delay  : {2}\n weight : {3}".format(
                self._pre,
                self._post,
                self._d,
                self._w,
            )
        )

    def __str__(self):
        return "Synapse {0}({1}, {2})".format(self.name, self._d, self._w)

    def __repr__(self):
        return "s_" + self._pre.name + "_" + self._post.name

    def update_state(self):
        """
        updates the time evolution of the states for one time step. The spike information is
        sent through a queue of length given by the delay and scaled by the weight value.

        Returns:
            None
        """

        if self._pre.spike:
            self._hist.append(self._w)
        else:
            self._hist.append(0.0)

        self._hist.popleft()


if __name__ == "__main__":
    from fugu.simulators.SpikingNeuralNetwork.neuron import LIFNeuron

    n1 = LIFNeuron("n1")
    n2 = LIFNeuron("n2")
    s = Synapse(n1, n2, delay=1, weight=1.0)

    try:
        s.delay(0)
    except:
        print("Raised Value error Exception since delay is < 1")

    try:
        s.delay(2.5)
    except:
        print("Raised type error since delay was a float")

    print(f"Synapse parameters: {s.show_params()}")
    try:
        s.set_params(new_delay=-1)
    except:
        print("Raised Value error Exception since delay is < 1")

    s.set_params(2, 2.0)
    s.show_params()
