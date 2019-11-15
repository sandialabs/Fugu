#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 14:46:55 2019

@author: smusuva
"""
import abc
import sys
import numpy as np

from abc import abstractmethod
from collections import deque
from warnings import warn

if sys.version_info >= (3, 4):
    ABC = abc.ABC
else:
    ABC = abc.ABCMeta('ABC', (), {'__slots__': ()})

default_brick_metadata = {
                           'input_shape': [()],
                           'output_shape': [()],
                           'D': 0,
                           'layer': 'output',
                           'input_coding': 'unknown',
                           'output_coding': 'unknown',
                           }

input_coding_types = [
                       'current',
                       'unary-B',
                       'unary-L',
                       'binary-B',
                       'binary-L',
                       'temporal-B',
                       'temporal-L',
                       'Raster',
                       'Population',
                       'Rate',
                       'Undefined',
                       ]


class Brick(ABC):
    """
    Abstract Base Class definition of a Brick class
    """

    def __init__(self):
        self.name = "Empty Brick"
        self.is_built = False
        self.supported_codings = []

    @abstractmethod
    def build(self, graph, metadata, control_nodes, input_lists, input_codings):
        """
        Build the computational graph of the brick. Method must be defined in any class inheriting from Brick.

        Arguments:
            + graph - networkx graph
            + metadata - A dictionary of shapes and parameters
            + control_nodes - list of dictionary of auxillary nodes.
                Acceptable keys include:
                    'complete' - A list of neurons that fire when the brick is done
                    'begin' - A list of neurons that fire when the brick begins computation
                                (used for temporal processing)
            + input_lists - list of lists of nodes for input neurons
            + input_codings - list of input coding types (as strings)
        """
        pass


class InputBrick(Brick):
    """
    Abstract Base class for handling inputs inherited from Brick
    """

    def __init__(self):
        self.streaming = False
        self.is_multi_runs = False

    @abstractmethod
    def __iter__(self):
        pass

    @abstractmethod
    def __next__(self):
        pass

    @abstractmethod
    def get_input_value(self, t=None):
        """
        Abstract method to get input values. InputBricks must implement this method

        Arguments:
            + t - type of input (Default: None)
        """
        pass


class InputSource:
    """
    Base class for handling various input sources/streams.

    Example: Converted output (as a stream of 0's and 1's) from a DVI camera
    """

    def __init__(self):
        self.name = "Empty Source"

    @abstractmethod
    def connect(self, graph, metadata, source):
        """
        Abstract method that tells the scaffold how it should connect the source to the circuit.
        This is accomplished by using a "source" dictionary argument when you create neurons/synapses.
        The "source" dictionary will contain whatever information the backends will need.

        Example 1: 
            Suppose the source was a motion detector (connected to hardware by a usb).
            Everytime there is movement detected, we want a specific neuron to fire.
            Then connect might look something like:
                connect():
                    scaffold.graph.add_node(
                                     "Sensor",
                                     threshold=1.0,
                                     decay=1.0,
                                     potential=0.0,
                                     source={
                                              'device_type': 'usb',
                                              'device_name': 'camera',
                                              'device_id': '00:07.0',
                                            },
                                     )
                    scaffold.graph.add_edge(
                                     "Sensor",
                                     some_other_neuron,
                                     weight=1.0,
                                     delay=1.0,
                                     )

        Example 2:
            Suppose the source was a network port receiving TCP packets
            We want to fire specific neurons based on which flag bits are set
            Then connect might look something like:
                connect():
                    scaffold.graph.add_node(
                                     "TCPPortA",
                                     threshold=1.0,
                                     decay=1.0,
                                     potential=0.0,
                                     source={
                                              'device_type': 'tcp_port',
                                              'device_id': '20',
                                              'flag_bit_id': 1,  # Look at flag bit #1 
                                              'flag_fire_value': 0,  # Fire if flag bit #1 is a 0
                                            },
                                     )
                    scaffold.graph.add_edge(
                                     "TCPPortA",
                                     some_other_neuron,
                                     weight=1.0,
                                     delay=1.0,
                                     )
                    scaffold.graph.add_node(
                                     "TCPPortB",
                                     threshold=1.0,
                                     decay=1.0,
                                     potential=0.0,
                                     source={
                                              'device_type': 'tcp_port',
                                              'device_id': '20',
                                              'flag_bit_id': 3,  # Look at flag bit #3 
                                              'flag_fire_value': 1,  # Fire if flag bit #3 is a 1
                                            },
                                     )
                    scaffold.graph.add_edge(
                                     "TCPPortB",
                                     some_other_neuron,
                                     weight=1.0,
                                     delay=1.0,
                                     )
                    scaffold.graph.add_node(
                                     "TCPPortC",
                                     threshold=1.0,
                                     decay=1.0,
                                     potential=0.0,
                                     source={
                                              'device_type': 'tcp_port',
                                              'device_id': '20',
                                              'flag_bit_id': 7,  # Look at flag bit #7 
                                              'flag_fire_value': 0,  # Fire if flag bit #7 is a 0
                                            },
                                     )
                    scaffold.graph.add_edge(
                                     "TCPPortC",
                                     some_other_neuron,
                                     weight=1.0,
                                     delay=1.0,
                                     )
        """
        pass


class Vector_Input(InputBrick):
    """
    Class to handle a vector of spiking input. Inherits from InputBrick
    """

    def __init__(
          self,
          spikes,
          time_dimension=False,
          coding='Undefined',
          batchable=True,
          name=None,
          multi_run_inputs=False,
          ):
        """
        Construtor for this brick.
        Arguments:
            + spikes - A numpy array of which neurons should spike at which times
            + time_dimension - Time dimesion is included as dimension -1
            + coding - Coding type to be represented.
            + batchable - True if input should represent static data; currently True is the only supported mode.
            + name - Name of the brick.  If not specified, a default will be used.  Name should be unique.
            + multi_run_inputs - True if 'spikes' represents inputs for different runs
        """
        super(InputBrick, self).__init__()
        self.vector = np.array(spikes)
        self.coding = coding
        self.time_dimension = time_dimension
        self.is_built = False
        self.batchable = batchable
        self.name = name
        self.index_map = None
        self.supported_codings = []
        self.metadata = {'D': 0}
        self.current_time = 0
        self.is_multi_runs = multi_run_inputs

    def __iter__(self):
        self.current_time = 0
        return self

    def __next__(self):
        if self.is_multi_runs:
            if self.current_time < len(self.vector):
                spiking_neurons = []
                for i, spike in enumerate(self.vector[self.current_time]):
                    if spike > 0:
                        spiking_neurons.append("{}_({},)".format(self.name, i))
                self.current_time += 1
                return spiking_neurons
            else:
                raise StopIteration
        else:
            if self.vector.shape[-1] > self.current_time:
                self.current_time += 1
                this_time_vector = self.vector[..., self.current_time - 1]
                local_idxs = np.array(np.where(this_time_vector))
                num_spikes = len(local_idxs[0])
                global_idxs = deque()
                for spike in range(num_spikes):
                    idx_to_build = deque()
                    for dimension in range(len(local_idxs)):
                        idx_to_build.append(local_idxs[dimension][spike])
                    global_idxs.append(tuple(idx_to_build))
                spiking_neurons = [self.name + "_" + str(idx) for idx in global_idxs]
                return spiking_neurons
            else:
                raise StopIteration

    next = __next__

    def get_input_value(self, t=None):
        warn("get_input_value is deprecated and will be removed from later versions.")
        warn("Please ensure that your backend is up-to-date.")
        if t is None:
            return self.vector
        else:
            assert type(t) is int
            return self.vector[..., t: t + 1][..., -1]

    def build(self, graph, metadata, control_nodes, input_lists, input_codings):
        """
        Build spike input brick.

        Arguments:
            + graph - networkx graph to define connections of the computational graph
            + metadata - dictionary to define the shapes and parameters of the brick
            + control_nodes - list of dictionary of auxillary nodes.
                Expected keys:
                    'complete' - A list of neurons that fire when the brick is done
            + input_lists - list of nodes that will contain input
            + input_coding - list of input coding formats

        Returns:
            + graph of a computational elements and connections
            + dictionary of output parameters (shape, coding, layers, depth, etc)
            + list of dictionary of control nodes ('complete')
            + list of output
            + list of coding formats of output
        """

        if not self.time_dimension and not self.is_multi_runs:
            self.vector = np.expand_dims(self.vector, len(self.vector.shape))

        output_lists = [[]]

        if self.is_multi_runs:
            temp_vector = self.vector[0]
            if not self.time_dimension:
                temp_vector = np.expand_dims(temp_vector, len(temp_vector.shape))
            self.index_map = np.ndindex(temp_vector.shape[:-1])
        else:
            self.index_map = np.ndindex(self.vector.shape[:-1])
        for i, index in enumerate(self.index_map):
            neuron_name = self.name + "_" + str(index)

            graph.add_node(neuron_name, index=index, threshold=0.0, decay=0.0, p=1.0)
            output_lists[0].append(neuron_name)
        output_codings = [self.coding]
        complete_node = self.name + "_complete"
        graph.add_node(complete_node, index=-1, threshold=0.0, decay=0.0, p=1.0, potential=0.5)
        self.is_built = True

        return (
                 graph,
                 {'output_shape': [self.vector.shape], 'output_coding': self.coding, 'layer': input, 'D': 0},
                 [{'complete': complete_node}],
                 output_lists,
                 output_codings,
                 )


class Spike_Input(Vector_Input):
    def __init__(self, input_spikes, *args, **kwargs):
        super().__init__(input_spikes, *args, **kwargs)
        warn("Spike_Input is deprecated.  Use Vector_Input instead.")
