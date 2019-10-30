#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 14:46:55 2019

@author: smusuva
"""
import abc
import sys
import numpy as np

if sys.version_info >= (3, 4):
    ABC = abc.ABC
else:
    ABC = abc.ABCMeta('ABC', (), {'__slots__': ()})

from abc import abstractmethod
from collections import deque
from warnings import warn

default_brick_metadata = {
      'input_shape' : [()],
      'output_shape' : [()],
      'D' : 0,
      'layer' : 'output',
      'input_coding' : 'unknown',
      'output_coding' : 'unknown',
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
    """Abstract Base Class definition of a Brick class"""

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
            + control_nodes - list of dictionary of auxillary nodes.  Acceptable keys include: 'complete' - A list of neurons that fire when the brick is done, 'begin' - A list of neurons that fire when the brick begins computation (used for temporal processing)
            + input_lists - list of lists of nodes for input neurons
            + input_codings - list of input coding types (as strings)
        """
        pass

class InputBrick(Brick):
    """Abstract Base class for handling inputs inherited from Brick"""
    
    def __init__(self):
        self.streaming = False
    
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
    
class Vector_Input(InputBrick):
    """Class to handle a vector of spiking input. Inherits from InputBrick"""

    def __init__(self, spikes, time_dimension=False, coding='Undefined', batchable=True, name=None):
        '''
        Construtor for this brick.
        Arguments:
		    + spikes - A numpy array of which neurons should spike at which times
			+ time_dimension - Time dimesion is included as dimension -1
			+ coding - Coding type to be represented.
            + batchable - True if input should represent static data; currently True is the only supported mode.
		    + name - Name of the brick.  If not specified, a default will be used.  Name should be unique.
        '''
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
        self.current_time=0
        
    def __iter__(self):
        self.current_time=0
        return self
    
    def __next__(self):
        if self.vector.shape[-1] > self.current_time:
            self.current_time += 1
            this_time_vector = self.vector[...,self.current_time-1]
            local_idxs = np.array(np.where(this_time_vector))
            num_spikes = len(local_idxs[0])
            global_idxs = deque()
            for spike in range(num_spikes):
                idx_to_build = deque()
                for dimension in range(len(local_idxs)):
                    idx_to_build.append(local_idxs[dimension][spike])
                global_idxs.append(tuple(idx_to_build))
            spiking_neurons = [self.name+"_"+str(idx) for idx in global_idxs]
            return spiking_neurons
        else:
            raise StopIteration

    next = __next__


    def get_input_value(self, t=None):
        warn("get_input_value is deprecated and will be removed from later versions.  Please ensure that your backend is up-to-date.")
        if t is None:
            return self.vector
        else:
            assert type(t) is int
            return self.vector[...,t:t+1][...,-1]

    def build(self, graph, metadata, control_nodes, input_lists, input_codings):
        """
        Build spike input brick.

        Arguments:
            + graph - networkx graph to define connections of the computational graph
            + metadata - dictionary to define the shapes and parameters of the brick
            + control_nodes - list of dictionary of auxillary nodes.  Excpected keys: 'complete' - A list of neurons that fire when the brick is done
            + input_lists - list of nodes that will contain input
            + input_coding - list of input coding formats

        Returns:
            + graph of a computational elements and connections
            + dictionary of output parameters (shape, coding, layers, depth, etc)
            + list of dictionary of control nodes ('complete')
            + list of output
            + list of coding formats of output
        """

        if not self.time_dimension:
            self.vector = np.expand_dims(self.vector,
                                         len(self.vector.shape))

        output_lists = [[]]

        self.index_map = np.ndindex(self.vector.shape[:-1])
        for i, index in enumerate(self.index_map):
            #neuron_name = self.name+"_" + str(i)
            neuron_name = self.name+"_"+str(index)

            graph.add_node(neuron_name,
                           index=index,
                           threshold=0.0,
                           decay=0.0,
                           p=1.0)
            output_lists[0].append(neuron_name)
        output_codings = [self.coding]
        complete_node = self.name+"_complete"
        #outputs = [{'neurons':output_lists,
        #            'shape':self.vector.shape[:-1],
        #            'coding':output_codings,
        #            'complete_node':complete_node}]
        graph.add_node(complete_node,
                      index = -1,
                      threshold = 0.0,
                      decay = 0.0,
                      p=1.0,
                      potential = 0.5)
        self.is_built = True
        return (graph, {'output_shape':[self.vector.shape], 'output_coding':self.coding, 'layer' : input, 'D':0},
               [{'complete':complete_node}], output_lists, output_codings)
        
class Spike_Input(Vector_Input):
    def __init__(self,input_spikes,*args,**kwargs):
        super().__init__(input_spikes, *args, **kwargs)
        warn("Spike_Input is deprecated.  Use Vector_Input instead.")
