#!/usr/bin/env python3
# isort: skip_file
import logging
import numpy as np
from .bricks import Brick
from .keras_utils import is_int_or_2d_tuple_of_ints, parse_thresholds_input_parameter, parse_strides_input_parameter

# Turn off black formatting for this file
# fmt: off

def parse_pool_size_input_parameter(pool_size_input):
    if is_int_or_2d_tuple_of_ints(pool_size_input):
        return pool_size_input if hasattr(pool_size_input, "__len__") else tuple(map(int, [pool_size_input, pool_size_input]))
    else:
        raise ValueError("'pool_size' must be an integer or tuple of 2 integers.")

class keras_pooling_2d_4dinput(Brick):
    'Pooling Layer brick'
    """
    Pooling Layer Function
    Michael Krygier
    mkrygie@sandia.gov
    
    """
    
    def __init__(self, pool_size, strides=None, thresholds=0.9, name=None, padding="same", method="max", layer_name='pooling', data_format="channels_last"):
        super().__init__()
        self.is_built = False
        self.name = name
        self.supported_codings = ['binary-L']

        self.pool_size = parse_pool_size_input_parameter(pool_size)
        self.strides = self.pool_size if strides is None else parse_strides_input_parameter(strides, error_message="'strides' must be an integer, tuple of 2 integers, or None. If None then defaults to 'pool_size'")

        self.padding = padding.lower()
        self.thresholds = thresholds
        self.method = method.lower()
        self.data_format = data_format.lower()
        self.metadata = {'pooling_size': self.pool_size, 'isKerasBrickLayer': True, 'layer_name': layer_name, 'padding': self.padding, 'strides': self.strides, 'method': self.method}
        
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
        self.input_shape = metadata[0].get('output_shape')
        assert self.input_shape is not None

        self.metadata['input_shape'] = self.input_shape
        self.initialize_input_shape_params()
        self.initialize_spatial_input_shape()
        self.initialize_spatial_output_shape()
        self.initialize_output_shape()
        self.metadata['output_shape'] = self.output_shape

        output_codings = [input_codings[0]]
        
        complete_node = self.name + "_complete"
        begin_node = self.name + "_begin"

        graph.add_node(begin_node   , index=-1, threshold=0.9, decay=1.0, p=1.0, potential=0.0)
        graph.add_node(complete_node, index=0 , threshold=0.9, decay=1.0, p=1.0, potential=0.0)

        graph.add_edge(control_nodes[0]["complete"], complete_node, weight=1.0, delay=1)
        graph.add_edge(control_nodes[0]["begin"]   , begin_node   , weight=1.0, delay=1)

        # Restrict max pooling threshold value to 0.9. Otherwise, the neuron circuit will not behave like an OR operation.
        if self.method == "max" and not np.any(np.array(self.thresholds) < 1.0):
            print("You cannot modify the threshold value for method='max' in pooling brick. You must use the default threshold value here.")
            raise ValueError(f"Max pooling requires a threshold value equal to 0.9.")
        
        # Check for scalar value for thresholds
        self.thresholds = parse_thresholds_input_parameter(self.thresholds, self.output_shape)
            
        # output neurons/nodes
        output_lists = [[]]
        for row in np.arange(0, self.spatial_output_shape[0]):
            for col in np.arange(0, self.spatial_output_shape[1]):
                for channel in np.arange(self.nChannels):
                    graph.add_node(f'{self.name}p{channel}_{row}_{col}', index=(row,col,channel), threshold=self.thresholds[0,row,col,channel], decay=1.0, p=1.0, potential=0.0)
                    output_lists[0].append(f'{self.name}p{channel}_{row}_{col}')

        # Collect Inputs
        pixels = np.reshape(input_lists[0], self.input_shape)

        edge_weights = self.get_edge_weights(self.pool_size, self.method)

        # Construct edges connecting input and output nodes
        # TODO: Handle cases where batch_size != 1; (i.e., index 0 in pixel[0,:,:,:] should not be hardcoded.)
        padded_input_shape_bounds = self.get_padded_input_shape_bounds(self.spatial_input_shape, self.pool_size, self.strides, self.spatial_output_shape)
        padded_row_stride_positions, padded_col_stride_positions = self.get_stride_positions_from_bounds(padded_input_shape_bounds)
        for output_row, padded_input_row in enumerate(padded_row_stride_positions[:self.spatial_output_shape[0]]):
            row_slice = slice(*self.get_adjusted_slice_positions(padded_input_row,self.spatial_input_shape[0],self.pool_size[0]))
            for output_col, padded_input_column in enumerate(padded_col_stride_positions[:self.spatial_output_shape[1]]):
                col_slice = slice(*self.get_adjusted_slice_positions(padded_input_column,self.spatial_input_shape[1],self.pool_size[1]))
                for channel in np.arange(self.nChannels):
                    # method 1
                    # for kx in np.arange(self.pool_size):
                    #     for ky in np.arange(self.pool_size):
                    #         graph.add_edge(pixels[0,rowpos+kx,colpos+ky], f'{self.name}p{channel}_{output_row}_{output_col}', weight=edge_weights, delay=1)
                    #         print(f" g{rowpos+kx}{colpos+ky} --> p{channel}_{output_row}_{output_col}")

                    # method 2
                    pixels_subset = pixels[0,row_slice,col_slice,channel]
                    for pixel in pixels_subset.flatten():
                        graph.add_edge(pixel, f'{self.name}p{channel}_{output_row}_{output_col}', weight=edge_weights, delay=1)
                        logging.debug(f" {pixel.split('_')[1]} --> p{channel}_{output_row}_{output_col}")

        self.is_built = True        
        return (graph, self.metadata, [{'complete': complete_node, 'begin': begin_node}], output_lists, output_codings)

    def get_adjusted_slice_positions(self, pos, input_length, pool_length):
        '''
            Returns the start and end slice positions from a specific position in the padded input row/column list
        '''
        if pos < 0:
            ipos = 0
            fpos = pos + pool_length
        elif pos + pool_length > input_length:
            ipos = input_length - pool_length + 1
            fpos = input_length
        else:
            ipos = pos
            fpos = pos + pool_length

        return ipos, fpos

    def get_edge_weights(self, kernel_shape, method):
        if method.lower() == "average":
            return 1.0 / np.prod(kernel_shape, dtype=float)
        elif method.lower() == "max":
            return 1.0
        else:
            raise ValueError("Undefined 'method' for edge weights.")

    def get_padded_input_shape_bounds(self, spatial_input_shape, kernel_shape, strides, spatial_output_shape):
        '''
            padding_shape = (prow,pcol)

            If the kernel height is odd then we will pad prow/2 rows on top and bottom. However, if the kernel height 
            is even then we pad floor(prow/2) rows on the top and ceil(prow/2) rows on the bottom. The width is padded
            in a similar fashion.
        '''
        padding_shape = self.get_padded_zeros_shape(spatial_input_shape, kernel_shape, strides, spatial_output_shape)
        assert self.check_padded_zeros_shape(padding_shape)

        # determine padding adjustments for top/bottom/left/right regions.
        top_bottom_padding = self.get_padding_amount(padding_shape[0])
        left_right_padding = self.get_padding_amount(padding_shape[1])

        # add adjustments to input_shape and return result
        top_bottom_bounds = [0 - top_bottom_padding[0], top_bottom_padding[1] + self.spatial_input_shape[1]]
        left_right_bounds = [0 - left_right_padding[0], left_right_padding[1] + self.spatial_input_shape[1]]
        return np.array([top_bottom_bounds, left_right_bounds]).astype(int)

    def get_padding_amount(self, dim_length):
        adjustments_array = np.array([np.floor(0.5*dim_length), np.ceil(0.5*dim_length)])
        return adjustments_array

    def get_padded_zeros_shape(self, spatial_input_shape, kernel_shape, strides_shape, spatial_output_shape):
        '''
            padding = strides*(output - 1) + kernel - input

            Returns the number (count) of padded zeros in the (rows,columns) necessary to reproduce the spatial
            output shape given the spatial input shape, kernel shape, and strides.
        '''
        spatial_input_shape = np.array(self.spatial_input_shape)
        kernel_shape = np.array(self.pool_size)
        strides_shape = np.array(self.strides)
        spatial_output_shape = np.array(self.spatial_output_shape)

        padding_count = (strides_shape*(spatial_output_shape - 1) + kernel_shape - spatial_input_shape)
        padding_count[padding_count < 0] = 0.
        return tuple(padding_count.astype(int))

    def check_padded_zeros_shape(self, padding_count):
        spatial_input_shape = np.array(self.spatial_input_shape)
        kernel_shape = np.array(self.pool_size)
        strides_shape = np.array(self.strides)
        spatial_output_shape = np.array(self.spatial_output_shape)
        padding_shape = np.array(padding_count)
        calculated_output_shape = np.floor( (spatial_input_shape - kernel_shape + padding_shape + strides_shape) / strides_shape)
        expected_output_shape = spatial_output_shape
        isSameOutputShape = expected_output_shape == calculated_output_shape
        return isSameOutputShape.all()

    def get_stride_positions_from_bounds(self, input_shape_bounds):
        return [np.arange(input_shape_bounds[0,0],input_shape_bounds[0,1],self.strides[0]),
                np.arange(input_shape_bounds[1,0],input_shape_bounds[1,1],self.strides[1])]

    def initialize_output_shape(self):
        if self.data_format == "channels_last":
            self.output_shape = (self.batch_size, *self.spatial_output_shape, self.nChannels)
        elif self.data_format == "channels_first":
            self.output_shape = (self.batch_size, self.nChannels, *self.spatial_output_shape)

    def initialize_spatial_input_shape(self):
        self.spatial_input_shape = (self.image_height, self.image_width)

    def initialize_input_shape_params(self):
        self.batch_size, self.image_height, self.image_width, self.nChannels = self.get_input_shape_params(self.input_shape)

    def get_spatial_input_shape(self, input_shape):
        batch_size, image_height, image_width, nChannels = self.get_input_shape_params(input_shape)
        spatial_input_shape = (image_height, image_width)
        return spatial_input_shape

    def get_input_shape_params(self, input_shape):
        if self.data_format == "channels_last":
            batch_size, image_height, image_width, nChannels = input_shape
        elif self.data_format == "channels_first":
            batch_size, nChannels, image_height, image_width = input_shape
        else:
            raise ValueError(f"'data_format' is either 'channels_first' or 'channels_last'. Received {self.data_format}")

        return batch_size, image_height, image_width, nChannels

    def get_same_padding_spatial_output_shape(self, spatial_input_shape, strides):
        return np.floor((np.array(spatial_input_shape) - 1) / np.array(strides)) + 1

    def get_valid_padding_spatial_output_shape(self, spatial_input_shape, kernel_shape, strides):
        return np.floor((np.array(spatial_input_shape) - np.array(kernel_shape)) / np.array(strides)) + 1

    def initialize_same_padding_spatial_output_shape(self):
        self.spatial_output_shape = list(map(int, self.get_same_padding_spatial_output_shape(self.spatial_input_shape, self.strides)))

    def initialize_valid_padding_spatial_output_shape(self):
        self.spatial_output_shape = list(map(int, self.get_valid_padding_spatial_output_shape(self.spatial_input_shape, self.pool_size, self.strides)))

    def get_spatial_output_shape(self, spatial_input_shape, kernel_shape, strides):
        if self.padding == "same":
            spatial_output_shape = self.get_same_padding_spatial_output_shape(spatial_input_shape, strides)
        elif self.padding == "valid":
            spatial_output_shape = self.get_valid_padding_spatial_output_shape(spatial_input_shape, kernel_shape, strides)
        else:
            raise ValueError(f"'pool_padding' is one of 'same' or 'valid'. Received {self.padding}.")

        spatial_output_shape = list(map(int,spatial_output_shape))
        return spatial_output_shape

    def initialize_spatial_output_shape(self):
        if self.padding == "same":
            self.initialize_same_padding_spatial_output_shape()
        elif self.padding == "valid":
            self.initialize_valid_padding_spatial_output_shape()
        else:
            raise ValueError(f"'pool_padding' is one of 'same' or 'valid'. Received {self.padding}.")
