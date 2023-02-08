#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from .bricks import InputBrick, input_coding_types

import numpy as np

from abc import abstractmethod
from collections import deque
from warnings import warn

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


class InputSource:
    """
    Base class for handling various input sources/streams.
    Example:
        Converted output (as a stream of 0's and 1's) from a DVI camera
    """
    def __init__(self):
        self.name = "InputSource"

    @abstractmethod
    def connect(self, graph, metadata, source):
        """
        Abstract method that tells the scaffold how it should connect the source to the circuit.
        This is accomplished by using a "source" dictionary argument when you create neurons/synapses.
        The "source" dictionary will contain whatever information the backends will need.
        Args:
            graph: networkx graph to define connections of the computational graph
                * If the graph has edge weights, this brick will solve the single source shortest paths problem
            metadata (dict): dictionary to define the shapes and parameters of the brick

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
    Construtor for this brick.
        Args:
            spikes (array): A numpy array of which neurons should spike at which times
            time_dimension: Time dimesion is included as dimension -1
            coding: Coding type to be represented.
            batchable: True if input should represent static data; currently True is the only supported mode.
            name (str): Name of the brick.  If not specified, a default will be used.  Name should be unique.
    """
    def __init__(
        self,
        spikes,
        time_dimension=False,
        coding='Undefined',
        batchable=True,
        name="VectorInput",
    ):

        super(Vector_Input, self).__init__(name)
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

    def __iter__(self):
        self.current_time = 0
        return self

    def __next__(self):
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
            spiking_neurons = [
                self.generate_neuron_name(str(idx)) for idx in global_idxs
            ]
            return spiking_neurons
        else:
            raise StopIteration

    next = __next__

    def set_properties(self, properties={}):
        new_vector = np.array(properties['spike_vector'])
        if 'time_dimension' not in properties or properties['time_dimension']:
            new_vector = np.expand_dims(new_vector, len(new_vector.shape))
        if new_vector.shape != self.vector.shape:
            raise ValueError(
                "Dimensions of new spike vector ({}) does not match expected ({})"
                .format(
                    new_vector.shape,
                    self.vector.shape,
                ), )
        else:
            self.vector = new_vector
            self.current_time = 0
        return None

    def get_input_value(self, t=None):
        warn(
            "get_input_value is deprecated and will be removed from later versions."
        )
        warn("Please ensure that your backend is up-to-date.")
        if t is None:
            return self.vector
        else:
            assert type(t) is int
            return self.vector[..., t:t + 1][..., -1]

    def build(self, graph, metadata, control_nodes, input_lists,
              input_codings):
        """
        Build spike input brick.

        Args:
            graph: networkx graph to define connections of the computational graph
            metadata: dictionary to define the shapes and parameters of the brick
            control_nodes: list of dictionary of auxillary nodes.
                Expected keys:
                    'complete' - A list of neurons that fire when the brick is done
            input_lists: list of nodes that will contain input
            input_coding: list of input coding formats

        Returns:
            graph: graph of a computational elements and connections
            output_shape, output_coding, layer, D: dictionary of output parameters (shape, coding, layers, depth, etc)
            complete_node, begin_node: list of dictionary of control nodes ('complete')
            output_lists (list): list of output
            output_codings (list): list of coding formats of output

        """

        if not self.time_dimension:
            self.vector = np.expand_dims(self.vector, len(self.vector.shape))

        complete_node = self.generate_neuron_name("complete")
        begin_node = self.generate_neuron_name("begin")
        vector_size = len(self.vector) * len(self.vector.shape)
        graph.add_node(begin_node,
                       index=-1,
                       threshold=0.0,
                       decay=0.0,
                       p=1.0,
                       potential=0.1)
        time_length = self.vector.shape[-1]
        if time_length == 1:
            graph.add_node(complete_node,
                           index=-1,
                           threshold=0.0,
                           decay=0.0,
                           p=1.0,
                           potential=0.1)
        else:
            graph.add_node(complete_node,
                           index=-1,
                           threshold=0.5,
                           decay=0.0,
                           p=1.0,
                           potential=0.0)
            graph.add_edge(begin_node,
                           complete_node,
                           weight=1.0,
                           delay=time_length - 1)

        output_lists = [[]]
        self.index_map = np.ndindex(self.vector.shape[:-1])
        for i, index in enumerate(self.index_map):
            neuron_name = self.generate_neuron_name(str(index))

            graph.add_node(neuron_name,
                           index=index,
                           threshold=0.0,
                           decay=0.0,
                           p=1.0)
            output_lists[0].append(neuron_name)
        output_codings = [self.coding]

        self.is_built = True

        return (
            graph,
            {
                'output_shape': [self.vector.shape],
                'output_coding': self.coding,
                'layer': input,
                'D': 0
            },
            [{
                'complete': complete_node,
                'begin': begin_node
            }],
            output_lists,
            output_codings,
        )

def _convert_base_p(arr, p=2, bits=8, time_dimension = False, collapse_binary=True):
    if p > 2:
        collapse_binary = False
    max_value = np.max(arr)
    if max_value >= p**bits - 1:
        raise ValueError("Max value of {} exceeded.".format(p**bits - 1))
    min_value = np.min(arr)
    if min_value < 0:
        raise ValueError("Inputs should be positive.")
    x = None
    for bit_no in range(bits):
        x_i = None
        mod = np.mod(np.floor(arr/(p**bit_no)), p)
        for p_i in range(p):
            this_slice = (mod==p_i).astype(np.int8)
            this_slice = np.expand_dims(this_slice, -1)
            this_slice = np.expand_dims(this_slice, -1)
            if p_i == 0:
                x_i = this_slice
            else:
                x_i = np.concatenate([x_i,this_slice], axis=-1)
        if collapse_binary:
            x_i = x_i[...,1]
        if bit_no == 0:
            x = x_i
        else:
            concat_axis = -1 if collapse_binary else -2
            x = np.concatenate([x, x_i], axis=concat_axis)
    if time_dimension:
        time_axis = -2 if collapse_binary else -3
        x = np.moveaxis(x, time_axis, -1)
    return x



class BaseP_Input(Vector_Input):
    """
    Class to handle converting a numpy array to a "Base p" encoding. Inherits from Vector_Input.

    By "Base p" we mean that an array is converted so that the numbers are represented by spikes using a particular base.

    The case where p=2 corresponds to binary.  (Note that binary has a specific collapse_binary described below.)
    
    For now, the brick supports converting non-negative integer values only.

    Here are some examples of how values are converted.

    x = 222  p=7  bits=3 ->  [5, 3, 4] as 214 = 5*7^0 + 3*7^1 + 4*7^2
    x = 222  p=3  bits=5 ->  [0,2,0,2,2] as 222 = 0*3^0 + 2*3^1 + 0*3^2 + 2*3^3 + 2*3^4

    The resulting values are unary coded, with the first entry corresponding to 0.  Neurons that correspond to the 
    factor will spike.
    (p=7) [5, 3, 4] = [[0,0,0,0,0,1,0], [0,0,0,1,0,0,0], [0,0,0,0,1,0,0]]
    (p=3) [0, 2, 0, 2, 2] = [[1,0,0], [0,0,1], [1,0,0], [0,0,1], [0,0,1]]

    Note that the input space will require N*p*bits input neurons (where N is the number of values in the array).

    In this configuration, there will be N*p spikes. 

    In general, the array becomes formatted as (..., p, bits).

    For p=2 (that is, binary), we have the option of collapse_binary, which removes one axis of redundant information.

    In this case, we map the last axis as 
    [1,0] -> 0
    [0, 1] -> 1

    and our array has shape (..., p).

    Lastly, we allow for a time_dimension.  If time_dimension is True, then the last dimension of the input 
    array is treated as a time index.  This is the same behavior as in the base Vector_Input. 

    If time_dimension=False, then all the spikes delivered at the same time (first timestep).

    If time_dimension=True, then the last dimension is treated as a time dimension and each slice creates spikes at that timestep.

    In this case, the last dimension is preserved as a time dimension.

    Examples:

    Input is (2,4), p=3, bits=5, time_dimension=False.  Output will be (2, 4, 3, 5) all of which are delivered at the first timestep.
    Input is (2,4), p=3, bits=5, time_dimension=True. Output will be (2,3,5) and there will be 4 timesteps of input spikes.


    Construtor for this brick.
        Args:
            array (ndarray): A numpy array of values.  Currently supported values are non-negative integers.
            p (int): Base to use
            bits (int): Number of 'bits' or factors to use
            collapse_binary (boolean): If true, binary arrays are collapsed to a single dim.
            time_dimension: Time dimesion is included as dimension -1
            coding: Coding type to be represented.
            name (str): Name of the brick.  If not specified, a default will be used.  Name should be unique.
    """
    def __init__(
        self,
        array,
        p=2,
        bits=8,
        collapse_binary=True,
        time_dimension=False,
        coding='Undefined',
        name="BasePInput",
    ):
        super(BaseP_Input, self).__init__([], 
                                    time_dimension = time_dimension,
                                    coding=coding,
                                    batchable=True,
                                    name=name)
        formatted_array =  _convert_base_p(array, 
                                            p=p,
                                            bits=bits,
                                            time_dimension = time_dimension,
                                            collapse_binary = collapse_binary)
        self.vector = formatted_array
        self.coding = 'binary_L' if p==2 else 'Undefined'
