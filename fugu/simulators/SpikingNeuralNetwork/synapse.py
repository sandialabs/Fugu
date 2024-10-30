#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import abc
import sys
from collections import deque
import sys
import numpy as np
from fugu.simulators.SpikingNeuralNetwork.learning_params import LearningParams

from fugu.simulators.SpikingNeuralNetwork.neuron import Neuron
from fugu.utils.types import float_types, int_types, str_types
from fugu.utils.validation import int_to_float, validate_type

if sys.version_info >= (3, 4):
    ABC = abc.ABC
else:
    ABC = abc.ABCMeta("ABC", (), {"__slots__": ()})


class Synapse(ABC):
    """
    Abstract base class for Synapses. This defines the interface for synapses with a minimum set of properties
    """

    @abc.abstractmethod
    def __init__(
        self, pre_neuron: Neuron = None, post_neuron: Neuron = None, weight: float = 0.0
    ):
        """
        Parameters:
            pre_neuron (Neuron): Pre-synaptic neuron that provides input to the synapse
            post_neuron (Neuron): Post-synaptic neuron
            weight (double): Synaptic weight
        """
        self.w_hist_trace = 0

    @abc.abstractmethod
    def update_state(self):
        """
        Update the state of the synapse
        """
        pass

    @abc.abstractmethod
    def get_key(self):
        """
        Get the key which is the pre and post synaptic neuron for the given synapse
        """

    @property
    @abc.abstractmethod
    def pre_neuron(self):
        """
        Getter for pre_neuron.
        """
        pass


class LearningSynapse(Synapse):
    """
    Synapses connect neurons in a neural network. The synapse class in a
    spiking objects models a simple synapse type that scales the input by a
    weight (double) and relays the information with a delay (non-negative int) of n
    time-steps.

    Parameters:
        Pre_neuron (Neuron): Pre-synaptic neuron that provides input to the synapse
        Post_neuron (Neuron): Post-synaptic neuron that receives the signals from the synapse
        learning_rule (str): optional.  The learning rule to be applied to the synapse. Currently provides options among (["STDP", "three-factor", "None"]). The default is None.
        Delay (int) : non-negative Int, optional.  Number of time steps needed to relay the scaled spike. The default is 1.
        Weight (double): optional.  Scaling value for incoming spike. The default is 1.0.
        Mod_neuron (Neuron): optional. Modulatory neuron for transmitting either errors or other modulatory signals
    """

    def __init__(
        self,
        pre_neuron: Neuron,
        post_neuron: Neuron,
        learning_rule: str = None,
        delay: int = 1,
        weight: float = 1.0,
        mod_neuron: Neuron = None,
        learning_params = None
    ):
        if not isinstance(pre_neuron, Neuron) or not isinstance(post_neuron, Neuron):
            raise TypeError("Pre and Post Synaptic neurons must be of type Neuron")
        if learning_rule is None:
            learning_rule = "None"
        weight = int_to_float(weight)
        validate_type(delay, int_types)
        validate_type(weight, float_types)
        validate_type(learning_rule, str_types)

        if delay < 1:
            raise ValueError("delay must be a strictly positive (>0) int value")

        if learning_rule not in ["STDP", "three-factor", "None"]:
            raise ValueError(
                "Learning rule must be one of the following: ['STDP', 'three-factor', 'None']"
            )

        super(LearningSynapse, self).__init__()
        self._d = delay
        self._w = weight
        self._pre = pre_neuron
        self._post = post_neuron
        self._hist = deque(np.zeros(self._d))
        self.name = "s_" + self._pre.name + "_" + self._post.name
        self._learning_rule = learning_rule
<<<<<<< HEAD
        self._name_learning_rule = self._learning_rule if learning_rule != "None" else "Simple"
=======
        self._name_learning_rule = (
            self._learning_rule if learning_rule != "None" else "Simple"
        )
>>>>>>> c607e12 (Fixing more pytest issues and cleaning up redundant code)
        if learning_rule == "three_factor":
            if not isinstance(mod_neuron, Neuron):
                raise TypeError("Modulatory neuron must be of type neuron")
            self._mod = mod_neuron
        self._learning_params = learning_params if learning_params is not None else LearningParams()
        self._eligibility_trace = 0.0

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
            self._pre (Neuron): Pre-synaptic neuron that provides input to the synapse
        """
        return self._pre

    @property
    def post_neuron(self):
        """
        Getter for post_neuron.
        Returns:
            self._post (Neuron): Post-synaptic neuron that receives the signals from the synapse
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
        delay, weight and the learning rule).

        Returns:
            None
        """

        print(
            "Synapse {0} -> {1}:\n delay  : {2}\n weight : {3}\n learning_rule : {4}".format(
                self._pre, self._post, self._d, self._w, self._learning_rule
            )
        )

    def __str__(self):
        return "{0}_Synapse {1}({2}, {3})".format(
            self._name_learning_rule, self.name, self._d, self._w
        )

    def __repr__(self):
        return "s_" + self._pre.name + "_" + self._post.name

    def get_learning_params(self):
        """
        Get the learning parameters for the synapse

        Returns:
            self._learning_params (LearningParams.__dict__): Learning parameters for the synapse as a dictionary
        """
        self._learning_params = LearningParams()
        return self._learning_params.__dict__

    def update_state(self):
        """
        Updates the time evolution of the states for one time step. The spike information is
        sent through a queue of length given by the delay and scaled by the weight value.

        Returns:
            None
        """
        if self._learning_rule != None:
            self.apply_learning()

            # TODO introduce a trace variable instead of the weight history
            # self.w_hist_trace.append(self._w)
            # print ("Entered this condition", self._pre.spike_hist, self._pre.spike)

        if self._pre.spike:
            self._hist.append(self._w)
        else:
            self._hist.append(0.0)

        self._hist.popleft()

    @staticmethod
    def calculate_spike_timing(spike_hist: list) -> int:
        """
        Calculate the timing between the reference and either pre or post synaptic spike

        Parameters:
            spike_hist (deque): Spike history of the post or pre synaptic neuron
        Returns:
            int: time difference between the pre and post synaptic spikes
        """
        spike_hist = list(map(int, spike_hist))
        if all(val == 0 for val in spike_hist):
            return 0
        else:
            spike_ind = max([ind for ind, val in enumerate(spike_hist) if val])
            spike_time = len(spike_hist) - spike_ind
            return spike_time

    def apply_learning(self) -> None:
        """
        Modify the weight of the synapse with STDP learning rule.
        The STDP learning rule is implemented as follows:

        When a post synaptic spike occurs after a pre synaptic spike, the weight is increased according to
        w += A_p * exp(-delta_t/tau), where delta_t is the difference in spike time of the post and pre synaptic neuron respectively

        When a post synaptic spike occurs before a pre synaptic spike, the weight is decreased according to
        w += A_n * exp(delta_t/tau), where delta_t is the difference in spike time of the post and pre synaptic neuron respectively

        Returns:
            None
        """

        pre_spike_hist = self._pre.spike_hist
        post_spike_hist = self._post.spike_hist

        if len(pre_spike_hist) <= self._d:
            self._w += 0
            return
            
        delay_shift = len(pre_spike_hist) - self._d - 1
        assert delay_shift >= 0, "Delay is greater than the length of the spike history"
        post_spike_time = self.calculate_spike_timing(post_spike_hist[:-1]) if len(post_spike_hist) > 1 else 0  
        assert post_spike_time >= 0, "Post spike time is negative"
        pre_spike_time = self.calculate_spike_timing(pre_spike_hist[: delay_shift + 1])
        
        # TODO Instead of implementing the learning process here, we can just call the learning rule class
        if self._learning_rule == "STDP":
            if self._post.spike:
                # If the pre synaptic neuron spikes at the same time step considering the delay
                if pre_spike_hist[delay_shift : delay_shift + 1][0]:
                    # Increase the weight
                    self._w += self._learning_params.A_p

                else:
                    # Increase the weight as post happened after pre
                    if pre_spike_time != 0:
                        self._w += self._learning_params.A_p * np.exp(
                            (-pre_spike_time) / self._learning_params.tau
                        )
                    else:
                        self._w += 0

            # If the post synaptic neuron does not spike
            else:
                # If the pre synaptic spike occurs at the current time step, then pre after post
                if pre_spike_hist[delay_shift : delay_shift + 1][0]:

                    # Decrease the weight as pre happened after post
                    if post_spike_time != 0: 
                        self._w += self._learning_params.A_n * np.exp(
                            (-post_spike_time) / self._learning_params.tau
                        )
                    else:
                        self._w += 0
                else:
                    self._w += 0

        elif self._learning_rule == "three-factor":
            pass

        else:
            pass


if __name__ == "__main__":
    from fugu.simulators.SpikingNeuralNetwork.neuron import LIFNeuron

    n1 = LIFNeuron("n1")
    n2 = LIFNeuron("n2")
    s = LearningSynapse(n1, n2, delay=1, weight=1.0)

    try:
        s.delay = 0
    except:
        print("Raised Value error Exception since delay is < 1")

    try:
        s.delay = 2.5
    except:
        print("Raised type error since delay was a float")

    print(f"Synapse parameters: {s.show_params()}")
    try:
        s.set_params(new_delay=-1)
    except:
        print("Raised Value error Exception since delay is < 1")

    s.set_params(2, 2.0)
    s.show_params()
