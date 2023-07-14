#!/usr/bin/env python3

import logging
import numpy as np
from .bricks import Brick

class pooling_1d(Brick):
    'Pooling Layer brick'
    """
    Pooling Layer Function
    Michael Krygier
    mkrygie@sandia.gov
    
    """
    
    def __init__(self, pool_size, strides=2, thresholds=0.9, name=None, method="max"):
        super().__init__()
        self.is_built = False
        self.name = name
        self.supported_codings = ['binary-L']
        self.pool_size = pool_size
        self.strides = strides
        self.thresholds = thresholds
        self.method = method
        self.metadata = {'pooling_size': pool_size, 'pooling_strides': strides, 'pooling_method': method}
        
    def build(self, graph, metadata, control_nodes, input_lists, input_codings):
        """
        Build Pooling Layer brick. 

        Arguments:
            + graph - networkx graph to define connections of the computational graph
            + metadata - dictionary to define the shapes and parameters of the brick
            + control_nodes - dictionary of lists of auxillary networkx nodes.  Expected keys: 'complete' - A list of neurons that fire when the brick is done
            + input_lists - list of nodes that will contain input
            + input_coding - list of input coding formats

        Returns:
            + graph of a computational elements and connections
            + dictionary of output parameters (shape, coding, layers, depth, etc)
            + dictionary of control nodes ('complete')
            + list of output
            + list of coding formats of output
        """
        if type(metadata) is list:
            self.metadata = {**metadata[0], **self.metadata}
        else:
            self.metadata = {**metadata, **self.metadata}

        self.input_shape = self.metadata['convolution_output_shape']
        self.metadata['pooling_input_shape'] = self.input_shape

        output_codings = [input_codings[0]]
        
        complete_node = self.name + '_complete'
        begin_node = self.name + '_begin'
        
        graph.add_node(begin_node   , index = -1, threshold = 0.0, decay =0.0, p=1.0, potential=0.0)
        graph.add_node(complete_node, index =  0, threshold = 0.9, decay =0.0, p=1.0, potential=0.0)
        
        graph.add_edge(control_nodes[0]['complete'], complete_node, weight=0.0, delay=1)
        graph.add_edge(control_nodes[0]['begin']   , begin_node   , weight=0.0, delay=1)

        num_input_neurons = len(input_lists[0])
        num_output_neurons = self.get_output_size(num_input_neurons, self.pool_size, self.strides)

        # Restrict max pooling threshold value to 0.9. Otherwise, the neuron circuit will not behave like an OR operation.
        if self.method.lower() == "max" and not np.any(np.array(self.thresholds) < 1.0):
            print("You cannot modify the threshold value for method='max' in pooling brick. You must use the default threshold value here.")
            raise ValueError(f"Max pooling requires a threshold value equal to 0.9.")
        
        # Check for scalar value for thresholds
        if not hasattr(self.thresholds, '__len__') and (not isinstance(self.thresholds, str)):
            self.thresholds = self.thresholds * np.ones(num_output_neurons)
        else:
            if self.thresholds.shape != (num_output_neurons,):
                raise ValueError(f"Threshold length {self.thresholds.shape} does not equal the output neuron length ({num_output_neurons},)."
                )
        self.metadata['pooling_output_shape'] = self.thresholds.shape

        # output neurons/nodes
        output_lists = [[]]
        for id in np.arange(num_output_neurons):
            graph.add_node(f'{self.name}p{id}', index=id, threshold=self.thresholds[id], decay=1.0, p=1.0, potential=0.0)
            output_lists[0].append(f'{self.name}p{id}')

        # Collect Inputs
        pixels = input_lists[0]

        edge_weights = 1.0
        if self.method == 'average':
            edge_weights = 1.0 / self.pool_size

        # Construct edges connecting input and output nodes
        stride_positions = self.get_stride_positions(num_input_neurons, self.pool_size, self.strides)
        for i in np.arange(num_output_neurons):  # loop over output neurons
            pos = stride_positions[i]
            for k in np.arange(self.pool_size):
                graph.add_edge(pixels[pos+k], f'{self.name}p{i}', weight=edge_weights, delay=1)
                print(f" g{pos+k} --> p{i}")

        self.is_built = True        
        return (graph, self.metadata, [{'complete': complete_node, 'begin': begin_node}], output_lists, output_codings)
    
    def get_output_size(self,pixel_size, pool_size, strides):
        return int(np.floor(1.0 + (np.float64(pixel_size) - pool_size)/strides))
        
    def get_stride_positions(self,pixel_size, pool_size, strides):
        return np.arange(0, pixel_size, strides, dtype=int)

class pooling_2d(Brick):
    'Pooling Layer brick'
    """
    Pooling Layer Function
    Michael Krygier
    mkrygie@sandia.gov
    
    """
    
    def __init__(self, pool_size, strides=2, thresholds=0.9, name=None, method="max"):
        super().__init__()
        self.is_built = False
        self.name = name
        self.supported_codings = ['binary-L']
        self.pool_size = pool_size
        self.strides = strides
        self.thresholds = thresholds
        self.method = method
        self.metadata = {'pooling_size': pool_size, 'pooling_strides': strides, 'pooling_method': method}
        
    def build(self, graph, metadata, control_nodes, input_lists, input_codings):
        """
        Build Pooling Layer brick. 

        Arguments:
            + graph - networkx graph to define connections of the computational graph
            + dimensionality - dictionary to define the shapes and parameters of the brick
            + control_nodes - dictionary of lists of auxillary networkx nodes.  Expected keys: 'complete' - A list of neurons that fire when the brick is done
            + input_lists - list of nodes that will contain input
            + input_coding - list of input coding formats

        Returns:
            + graph of a computational elements and connections
            + dictionary of output parameters (shape, coding, layers, depth, etc)
            + dictionary of control nodes ('complete')
            + list of output
            + list of coding formats of output
        """
        if type(metadata) is list:
            self.metadata = {**metadata[0], **self.metadata}
        else:
            self.metadata = {**metadata, **self.metadata}

        self.input_shape = self.metadata['convolution_output_shape']
        self.metadata['pooling_input_shape'] = self.input_shape

        output_codings = [input_codings[0]]
        
        complete_node = self.name + '_complete'
        begin_node = self.name + '_begin'
        
        graph.add_node(begin_node   , index = -1, threshold = 0.0, decay =0.0, p=1.0, potential=0.0)
        graph.add_node(complete_node, index =  0, threshold = 0.9, decay =0.0, p=1.0, potential=0.0)
        
        graph.add_edge(control_nodes[0]['complete'], complete_node, weight=0.0, delay=1)
        graph.add_edge(control_nodes[0]['begin']   , begin_node   , weight=0.0, delay=1)

        num_input_neurons = len(input_lists[0])

        # determine output neuron bounds based on "input length", "pool_size", and "strides"
        # floor(1 + [Am + 2*pad_length - Bm ] / stride)
        # Am = Convolution Output Length
        # Bm = pool_size
        # pad_length = 0 (padding is taken care in the convolution brick)
        self.output_shape = self.get_output_shape()
        self.metadata['pooling_output_shape'] = self.output_shape
        num_output_neurons = self.output_shape[0] * self.output_shape[1]


        # Restrict max pooling threshold value to 0.9. Otherwise, the neuron circuit will not behave like an OR operation.
        if self.method.lower() == "max" and not np.any(np.array(self.thresholds) < 1.0):
            print("You cannot modify the threshold value for method='max' in pooling brick. You must use the default threshold value here.")
            raise ValueError(f"Max pooling requires a threshold value equal to 0.9.")
        
        # Check for scalar value for thresholds
        if not hasattr(self.thresholds, '__len__') and (not isinstance(self.thresholds, str)):
            self.thresholds = self.thresholds * np.ones(self.output_shape)
        else:
            if self.thresholds.shape != self.output_shape:
                raise ValueError(f"Threshold length {self.thresholds.shape} does not equal the output neuron length ({num_output_neurons},)."
                )
            
        # output neurons/nodes
        output_lists = [[]]
        for row in np.arange(self.output_shape[0]):
            for col in np.arange(self.output_shape[1]):
                graph.add_node(f'{self.name}p{row}{col}', index=(row,col), threshold=self.thresholds[row,col], decay=1.0, p=1.0, potential=0.0)
                output_lists[0].append(f'{self.name}p{row}{col}')

        # Collect Inputs
        pixels = input_lists[0]
        pixels = np.reshape(input_lists[0], self.input_shape)

        edge_weights = 1.0
        if self.method == 'average':
            edge_weights = 1.0 / self.pool_size ** 2

        # Construct edges connecting input and output nodes
        row_stride_positions = self.get_stride_positions(self.input_shape[0])
        col_stride_positions = self.get_stride_positions(self.input_shape[1])
        for row in np.arange(self.output_shape[0]):
            rowpos = row_stride_positions[row]
            for col in np.arange(self.output_shape[1]):
                colpos = col_stride_positions[col]

                # method 1
                # for kx in np.arange(self.pool_size):
                #     for ky in np.arange(self.pool_size):
                #         graph.add_edge(pixels[rowpos+kx,colpos+ky], f'{self.name}p{row}{col}', weight=edge_weights, delay=1)
                #         print(f" g{rowpos+kx}{colpos+ky} --> p{row}{col}")    

                # method 2
                tmp = pixels[rowpos:rowpos+self.pool_size,colpos:colpos+self.pool_size]
                for pixel in tmp.flatten():
                    graph.add_edge(pixel, f'{self.name}p{row}{col}', weight=edge_weights, delay=1)
                    print(f" {pixel.split('_')[1]} --> p{row}{col}")                        

        self.is_built = True        
        return (graph, self.metadata, [{'complete': complete_node, 'begin': begin_node}], output_lists, output_codings)
    
    def get_stride_positions(self, pixel_dim):
        return np.arange(0, pixel_dim, self.strides, dtype=int)
    
    def get_output_size(self, pixel_dim):
        return int(np.floor(1.0 + (np.float64(pixel_dim) - self.pool_size)/self.strides))
    
    def get_output_shape(self):
        pixel_dim1, pixel_dim2 = self.input_shape
        return (self.get_output_size(pixel_dim1), self.get_output_size(pixel_dim2))