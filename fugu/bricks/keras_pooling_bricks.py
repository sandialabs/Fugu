#!/usr/bin/env python3
# isort: skip_file
import logging
import numpy as np
from .bricks import Brick

# Turn off black formatting for this file
# fmt: off

class keras_pooling_2d_4dinput(Brick):
    'Pooling Layer brick'
    """
    Pooling Layer Function
    Michael Krygier
    mkrygie@sandia.gov
    
    """
    
    def __init__(self, pool_size, strides=None, thresholds=0.9, name=None, padding="same", method="max", data_format="channels_last"):
        super().__init__()
        self.is_built = False
        self.name = name
        self.supported_codings = ['binary-L']

        if hasattr(pool_size,"__len__"):
            if type(pool_size) is not tuple:
                raise ValueError("'pool_size' must be an integer or tuple of 2 integers.")

            if len(pool_size) != 2:
                raise ValueError("'pool_size' must be an integer or tuple of 2 integers.")
        elif isinstance(pool_size,float):
            raise ValueError("'pool_size' must be an integer or tuple of 2 integers.")
        else:
            pool_size = (pool_size, pool_size)

        if strides is None:
            strides = pool_size
        elif hasattr(strides, "__len__"):
            if type(strides) is not tuple:
                raise ValueError("'strides' must be an integer, tuple of 2 integers, or None. If None then defaults to 'pool_size'")

            if len(strides) != 2:
                raise ValueError("'strides' must be an integer, tuple of 2 integers, or None. If None then defaults to 'pool_size'")
        elif isinstance(strides, float):
            raise ValueError("'strides' must be an integer, tuple of 2 integers, or None. If None then defaults to 'pool_size'")
        else:
            strides = (strides, strides)

        self.pool_size = pool_size
        self.padding = padding
        self.strides = strides
        self.thresholds = thresholds
        self.method = method
        self.data_format = data_format
        self.metadata = {'pooling_size': pool_size, 'pooling_strides': strides, 'pooling_padding': padding, 'pooling_method': method}
        
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
        
        complete_node = self.name + "_complete"
        begin_node = self.name + "_begin"

        graph.add_node(begin_node   , index=-1, threshold=0.9, decay=1.0, p=1.0, potential=0.0)
        graph.add_node(complete_node, index=0 , threshold=0.9, decay=1.0, p=1.0, potential=0.0)

        graph.add_edge(control_nodes[0]["complete"], complete_node, weight=1.0, delay=1)
        graph.add_edge(control_nodes[0]["begin"]   , begin_node   , weight=1.0, delay=1)

        # determine output neuron bounds based on "input length", "pool_size", and "strides"
        # floor(1 + [Am + 2*pad_length - Bm ] / stride)
        # Am = Convolution Output Length
        # Bm = pool_size
        # pad_length = 0 (padding is taken care in the convolution brick)
        self.output_shape = self.get_output_shape()
        self.metadata['pooling_output_shape'] = self.output_shape

        # Restrict max pooling threshold value to 0.9. Otherwise, the neuron circuit will not behave like an OR operation.
        if self.method.lower() == "max" and not np.any(np.array(self.thresholds) < 1.0):
            print("You cannot modify the threshold value for method='max' in pooling brick. You must use the default threshold value here.")
            raise ValueError(f"Max pooling requires a threshold value equal to 0.9.")
        
        # Check for scalar value for thresholds
        if not hasattr(self.thresholds, '__len__') and (not isinstance(self.thresholds, str)):
            self.thresholds = self.thresholds * np.ones(self.output_shape)
        else:
            if self.thresholds.shape != self.output_shape:
                raise ValueError(f"Threshold shape {self.thresholds.shape} does not equal the output neuron shape {self.output_shape}."
                )
            
        # output neurons/nodes
        output_lists = [[]]
        for row in np.arange(0, self.spatial_output_shape[0]):
            for col in np.arange(0, self.spatial_output_shape[1]):
                for channel in np.arange(self.nChannels):
                    graph.add_node(f'{self.name}p{channel}{row}{col}', index=(row,col,channel), threshold=self.thresholds[0,row,col,channel], decay=1.0, p=1.0, potential=0.0)
                    output_lists[0].append(f'{self.name}p{channel}{row}{col}')

        # Collect Inputs
        pixels = np.reshape(input_lists[0], self.input_shape)

        edge_weights = 1.0
        if self.method == 'average':
            edge_weights = 1.0 / np.prod(self.pool_size, dtype=float)

        # Construct edges connecting input and output nodes
        # TODO: Handle cases where batch_size != 1; (i.e., index 0 in pixel[0,:,:,:] should not be hardcoded.)
        padded_input_shape_bounds = self.get_padded_input_shape_bounds()
        row_stride_positions, col_stride_positions = self.get_stride_positions_from_bounds(padded_input_shape_bounds)
        for row, irow in enumerate(row_stride_positions[:self.spatial_output_shape[0]]):
            irowpos, frowpos = self.adjust_position_to_input_length(irow,self.spatial_input_shape[0],self.pool_size[0])
            for col, icol in enumerate(col_stride_positions[:self.spatial_output_shape[1]]):
                icolpos, fcolpos = self.adjust_position_to_input_length(icol,self.spatial_input_shape[1],self.pool_size[1])
                for channel in np.arange(self.nChannels):
                    # method 1
                    # for kx in np.arange(self.pool_size):
                    #     for ky in np.arange(self.pool_size):
                    #         graph.add_edge(pixels[0,rowpos+kx,colpos+ky], f'{self.name}p{channel}{row}{col}', weight=edge_weights, delay=1)
                    #         print(f" g{rowpos+kx}{colpos+ky} --> p{channel}{row}{col}")

                    # method 2
                    pixels_subset = pixels[0,irowpos:frowpos,icolpos:fcolpos,channel]
                    for pixel in pixels_subset.flatten():
                        graph.add_edge(pixel, f'{self.name}p{channel}{row}{col}', weight=edge_weights, delay=1)
                        logging.debug(f" {pixel.split('_')[1]} --> p{channel}{row}{col}")

        self.is_built = True        
        return (graph, self.metadata, [{'complete': complete_node, 'begin': begin_node}], output_lists, output_codings)

    def adjust_position_to_input_length(self, pos, input_length, pool_length):
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

    def get_padded_input_shape_bounds(self):
        '''
            padding_shape = (prow,pcol)

            If the kernel height is odd then we will pad prow/2 rows on top and bottom. However, if the kernel height 
            is even then we pad floor(prow/2) rows on the top and ceil(prow/2) rows on the bottom. The width is padded
            in a similar fashion.
        '''
        padding_shape = self.get_padded_zeros_count()
        assert self.check_padded_zeros_count(padding_shape)

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

    def get_padded_zeros_count(self):
        '''
            padding = strides*(output - 1) + kernel - input
        '''
        spatial_input_shape = np.array(self.spatial_input_shape)
        kernel_shape = np.array(self.pool_size)
        strides_shape = np.array(self.strides)
        spatial_output_shape = np.array(self.spatial_output_shape)

        padding_count = (strides_shape*(spatial_output_shape - 1) + kernel_shape - spatial_input_shape)
        padding_count[padding_count < 0] = 0.
        return tuple(padding_count.astype(int))

    def check_padded_zeros_count(self, padding_count):
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

    def get_output_shape(self):
        self.spatial_output_shape = self.get_spatial_output_shape()

        if self.data_format.lower() == "channels_last":
            output_shape = (self.batch_size, *self.spatial_output_shape, self.nChannels)
        else:
            output_shape = (self.batch_size, self.nChannels, *self.spatial_output_shape)

        return output_shape

    def get_spatial_input_shape(self):
        self.batch_size, self.image_height, self.image_width, self.nChannels = self.get_input_shape_params()
        spatial_input_shape = (self.image_height, self.image_width)
        return spatial_input_shape

    def same_padding_spatial_output_shape(self):
        if not hasattr(self,"spatial_input_shape"):
            self.spatial_input_shape = self.get_spatial_input_shape()

        return np.floor((np.array(self.spatial_input_shape) - 1) / np.array(self.strides)) + 1

    def valid_padding_spatial_output_shape(self):
        if not hasattr(self,"spatial_input_shape"):
            self.spatial_input_shape = self.get_spatial_input_shape()

        return np.floor((np.array(self.spatial_input_shape) - np.array(self.pool_size)) / np.array(self.strides)) + 1

    def get_spatial_output_shape(self):
        if self.padding.lower() == "same":
            spatial_output_shape = self.same_padding_spatial_output_shape()
        elif self.padding.lower() == "valid":
            spatial_output_shape = self.valid_padding_spatial_output_shape()
        else:
            raise ValueError(f"'pool_padding' is one of 'same' or 'valid'. Received {self.padding}.")

        spatial_output_shape = list(map(int,spatial_output_shape))
        return spatial_output_shape

    def get_input_shape_params(self):
        if self.data_format.lower() == "channels_last":
            batch_size, image_height, image_width, nChannels = self.input_shape
        elif self.data_format.lower() == "channels_first":
            batch_size, nChannels, image_height, image_width = self.input_shape
        else:
            raise ValueError(f"'data_format' is either 'channels_first' or 'channels_last'. Received {self.data_format}")

        return batch_size, image_height, image_width, nChannels
