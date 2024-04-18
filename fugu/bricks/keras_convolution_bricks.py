#!/usr/bin/env python3
# isort: skip_file
import logging
import numpy as np
from .bricks import Brick

# Turn off black formatting for this file
# fmt: off

def debug_input_index(input_lists, Am, An, basep, bits):
    for k in np.arange(len(input_lists[0])):
        row, col, pwr, Ck = np.unravel_index(k, (Am, An, bits, basep))
        print(f"{k:3d}  {np.mod(k,basep*bits):2d}  ({row},{col}) {Ck:2d}  {pwr:2d}")

def input_index_to_matrix_entry(input_shape,basep,bits,index):
    Am, An = input_shape
    linearized_index = np.ravel_multi_index(index,tuple(np.repeat([1,2,basep],[1,2,2])))# zero-based linearized index

    return np.unravel_index(linearized_index,(Am,An,basep*bits))[:2]

def isValueScalar(scalar):
    if not hasattr(scalar, '__len__') and (not isinstance(scalar, str)):
        return True
    else:
        return False

def create_array_from_scalar_value(scalar, shape):
    return scalar * np.ones(shape)

def correct_shape(variable, expected_variable_shape):
    if not type(variable) is np.ndarray:
        variable = np.array(variable)

    if variable.shape == expected_variable_shape:
        return True
    else:
        return False

class keras_convolution_2d_4dinput(Brick):
    'Convolution brick that assumes collapse_binary=False for all base-p values'
    """
    Convolution Function
    Michael Krygier
    mkrygie@sandia.gov

    """

    def __init__(self, input_shape, kernel, thresholds, basep, bits, name=None, mode='same', strides=(1,1), biases=None, data_format="channels_last"):
        # TODO: Add capability to handle Keras "data_format='channels_first'"
        super().__init__()
        self.is_built = False
        self.name = name
        self.supported_codings = ['binary-L']
        self.input_shape = input_shape # (batch size, height, width, nChannels), can assume batch size is 1 or None.
        self.kernel = np.array(kernel) # (kernel height, kernel width, nChannels, nFilters)
        self.basep = basep
        self.bits = bits
        self.padding = mode.lower()
        self.biases = biases   # shape must be (nFilters,)
        self.nFilters = self.kernel.shape[-1]

        self.data_format = data_format.lower()
        self.strides = self.parse_strides_input(strides)
        self.initialize_spatial_input_shape()
        self.initialize_kernel_shape()
        self.initialize_output_shape()
        self.initialize_output_bounds() # determine output neuron bounds based on the "padding/mode"
        self.thresholds = self.parse_thresholds_input(thresholds, self.output_shape)
        self.metadata = {'D': 2, 'basep': basep, 'bits': bits, 'convolution_padding': mode, 'convolution_input_shape': self.input_shape, 'convolution_strides': self.strides, 'convolution_output_shape': self.output_shape}

    def build(self, graph, metadata, control_nodes, input_lists, input_codings):
        """
        Build convolution brick.

        Arguments:
            + graph - networkx graph to define connections of the computational graph
            + metadata - dictionary to define the shapes and parameters of the brick
            + control_nodes - dictionary of lists of auxillary networkx nodes.  Excpected keys: 'complete' - A list of neurons that fire when the brick is done
            + input_lists - list of nodes that will contain input
            + input_coding - list of input coding formats

        Returns:
            + graph of a computational elements and connections
            + dictionary of output parameters (shape, coding, layers, depth, etc)
            + dictionary of control nodes ('complete')
            + list of output
            + list of coding formats of output
        """

        if len(input_codings) != 1:
            raise ValueError("convolution takes in 1 inputs of size n")

        output_codings = [input_codings[0]]

        complete_node = self.name + "_complete"
        begin_node = self.name + "_begin"

        graph.add_node(begin_node   , index=-2, threshold=0.9, decay=1.0, p=1.0, potential=0.0)
        graph.add_node(complete_node, index=-1, threshold=0.9, decay=1.0, p=1.0, potential=0.0)

        graph.add_edge(control_nodes[0]["complete"], complete_node, weight=1.0, delay=2)
        graph.add_edge(control_nodes[0]["begin"]   , begin_node   , weight=1.0, delay=2)

        output_lists = self.create_output_neurons(graph)
        self.create_biases_nodes_and_synapses(graph,control_nodes)
        self.connect_input_and_output_neurons(input_lists,graph)

        self.is_built=True

        return (graph, self.metadata, [{'complete': complete_node, 'begin': begin_node}], output_lists, output_codings)

    def parse_thresholds_input(self, thresholds, expected_shape):
        if isValueScalar(thresholds):
            return create_array_from_scalar_value(thresholds, expected_shape)
        else:
            thresholds = np.array(thresholds)
            if correct_shape(thresholds, expected_shape):
                return thresholds
            else:
                raise ValueError(f"Threshold shape {thresholds.shape} does not equal the output neuron shape {expected_shape}.")

    def parse_strides_input(self, strides):
        if isinstance(strides,(list,tuple)):
            if len(strides) > 2:
                raise ValueError("Strides must be an integer or tuple/list of 2 integers.")
            else:
                strides = tuple(map(int,strides))
        elif isinstance(strides,(float,int)):
            strides = tuple(map(int,[strides,strides]))
        else:
            raise ValueError("Check strides input variable.")
        
        return strides

    def create_output_neurons(self, graph):
        # output neurons/nodes
        output_lists = [[]]
        for ix, i in enumerate(np.arange(self.bnds[0,0],self.bnds[1,0] + 1,self.strides[0])):
            for jx, j in enumerate(np.arange(self.bnds[0,1],self.bnds[1,1] + 1,self.strides[1])):
                for kx in np.arange(self.nFilters):
                    graph.add_node(f'{self.name}g{kx}_{ix}_{jx}', index=(ix,jx,kx), threshold=self.thresholds[0,ix,jx,kx], decay=1.0, p=1.0, potential=0.0)
                    output_lists[0].append(f'{self.name}g{kx}_{ix}_{jx}')

        return output_lists

    def create_biases_nodes_and_synapses(self, graph, control_nodes):
        # Biases for convolution
        if self.biases is not None:
            # biases neurons/nodes; one node per kernel/channel in filter
            for k in np.arange(self.nFilters):
                graph.add_node(f'{self.name}b{k}', index=(99,k), threshold=0.9, decay=1.0, p=1.0, potential=0.0)
                graph.add_edge(control_nodes[0]["complete"], f'{self.name}b{k}', weight=1.0, delay=1)

            # Construct edges connecting biases node(s) to output nodes
            for k in np.arange(self.nFilters):
                for jx, j in enumerate(np.arange(self.bnds[0,0],self.bnds[1,0] + 1,self.strides[0])):
                    for ix, i in enumerate(np.arange(self.bnds[0,1],self.bnds[1,1] + 1,self.strides[1])):
                        graph.add_edge(f'{self.name}b{k}',f'{self.name}g{k}_{jx}_{ix}', weight=self.biases[k], delay=1)

    def connect_input_and_output_neurons(self,input_lists,graph):
        # Get size/shape information from input arrays
        batch_size, height, width, nChannels = self.input_shape
        Bm, Bn = self.kernel_shape

        num_input_neurons = len(input_lists[0])

        input_neurons_2_output_neurons = {(row,col): self.get_output_neurons(row,col) for col in np.arange(width) for row in np.arange(height)}
        # Collect Inputs
        I = np.array(input_lists[0])

        padded_input_shape_bounds = self.get_padded_input_shape_bounds(self.spatial_input_shape,self.kernel_shape,self.strides,self.spatial_output_shape)
        padded_row_stride_positions, padded_col_stride_positions = self.get_stride_positions_from_bounds(padded_input_shape_bounds)
        kernel_corner_position_on_padded_input_shape_bounds = [(a,b) for a in padded_row_stride_positions[:self.spatial_output_shape[0]] for b in padded_col_stride_positions[:self.spatial_output_shape[1]]]
        self.padded_top_row_position = padded_row_stride_positions[0]
        self.padded_left_col_position = padded_col_stride_positions[0]
        # Construct edges connecting input and output nodes
        cnt = -1
        for k in np.arange(num_input_neurons):  # loop over input neurons
            # input_row, input_col, channel, pwr, Ck = np.unravel_index(k, (height, width, self.nChannels, self.bits,  self.basep))
            try:
                input_row, input_col, channel, pwr, Ck = graph.nodes[I[k]]['index'][-5:]
                if Ck == 0:
                    continue
                constant = Ck * self.basep**pwr
            except ValueError:
                input_row, input_col, channel = graph.nodes[I[k]]['index'][-3:]
                constant = 1.0

            # loop over output neurons
            # TODO: Fix loop so that strides through memory are sequential, instead of jumping around in array 'kernel' indices
            for output_row, output_col in input_neurons_2_output_neurons[(input_row,input_col)]:
                ix, jx = self.get_kernel_indices(input_row,input_col,output_row,output_col)

                for filter in np.arange(self.nFilters):
                    cnt += 1
                    graph.add_edge(I[k], f'{self.name}g{filter}_{output_row}_{output_col}', weight=constant * self.kernel[ix,jx,channel,filter], delay=2)
                    # logging.debug(f'{cnt:3d}  A[m,n]: ({input_row:2d},{input_col:2d})   power: {pwr}    coeff_i: {Ck}    input: {k:3d}      output: {filter}{output_row}{output_col}   B[m,n]: ({ix:2d},{jx:2d})   filter: {self.kernel[ix,jx,channel,filter]}     I(row,col,channel,bit-pwr,basep-coeff): {np.unravel_index(k,(height,width,self.nChannels,self.bits,self.basep))}     I[index]: {graph.nodes[I[k]]["index"]}')

    def get_kernel_indices(self, input_row, input_col, output_row, output_col):
        ix = (self.kernel_shape[0]-1) - input_row + (self.padded_top_row_position + output_row * self.strides[0])
        jx = (self.kernel_shape[1]-1) - input_col + (self.padded_left_col_position + output_col * self.strides[1])
        return (ix,jx)

    def get_padded_input_shape_bounds(self, spatial_input_shape, kernel_shape, strides, spatial_output_shape):
        '''
            padding_shape = (prow,pcol)

            If the kernel height is odd then we will pad prow/2 rows on top and bottom. However, if the kernel height 
            is even then we pad floor(prow/2) rows on the top and ceil(prow/2) rows on the bottom. The width is padded
            in a similar fashion.
        '''
        padding_shape = self.get_padded_zeros_shape(spatial_input_shape,kernel_shape,strides,spatial_output_shape)
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

    def check_padded_zeros_shape(self, padding_count):
        spatial_input_shape = np.array(self.spatial_input_shape)
        kernel_shape = np.array(self.kernel_shape)
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

    def get_output_neurons(self,row,col):
        output_neuron_indices = []
        Bm, Bn = self.kernel_shape
        Sm, Sn = self.strides

        padded_input_shape_bounds = self.get_padded_input_shape_bounds(self.spatial_input_shape,self.kernel_shape,self.strides,self.spatial_output_shape)
        padded_row_stride_positions, padded_col_stride_positions = self.get_stride_positions_from_bounds(padded_input_shape_bounds)

        for outrow, krow in enumerate(padded_row_stride_positions[:self.spatial_output_shape[0]]):
            if row < krow or row >= krow + Bm:
                continue
            for outcol, kcol in enumerate(padded_col_stride_positions[:self.spatial_output_shape[1]]):
                if col < kcol or col >= kcol + Bn:
                    continue
                output_neuron_indices.append((outrow,outcol))

        return output_neuron_indices

    def initialize_output_bounds(self):
        spatial_input_shape = np.array(self.spatial_input_shape)
        kernel_shape = np.array(self.kernel_shape)
        full_output_shape = spatial_input_shape + kernel_shape - 1

        if self.padding == "same":
            lb = np.floor(0.5 * (full_output_shape - spatial_input_shape))
            ub = np.floor(0.5 * (full_output_shape + spatial_input_shape) - 1)
            self.bnds = np.array([lb, ub], dtype=int)

        if self.padding == "valid":
            lmins = np.minimum(spatial_input_shape, kernel_shape)
            lb = lmins - 1
            ub = np.array(full_output_shape) - lmins
            self.bnds = np.array([lb, ub], dtype=int) - 1

    def initialize_output_shape(self):
        self.initialize_spatial_output_shape()
        if self.data_format == "channels_last":
            self.output_shape = (self.batch_size, *self.spatial_output_shape, self.nFilters)
        elif self.data_format == "channels_first":
            self.output_shape = (self.batch_size, self.nFilters, *self.spatial_output_shape)

    def get_padded_zeros_shape(self, spatial_input_shape, kernel_shape, strides_shape, spatial_output_shape):
        '''
            padding = strides*(output - 1) + kernel - input

            Returns the number (count) of padded zeros in the (rows,columns) necessary to reproduce the spatial 
            output shape given the spatial input shape, kernel shape, and strides.
        '''        
        spatial_input_shape = np.array(spatial_input_shape)
        kernel_shape = np.array(kernel_shape)
        strides_shape = np.array(strides_shape)
        spatial_output_shape = np.array(spatial_output_shape)

        padding_count = (strides_shape*(spatial_output_shape - 1) + kernel_shape - spatial_input_shape)
        padding_count[padding_count < 0] = 0.
        return tuple(padding_count.astype(int))

    def get_padded_zeros_shape_alt(self, kernel_shape, strides_shape):
        '''
            padding_shape = (prow,pcol)
            kernel_shape = (krow,kcol)

            Setting (prow,pcol) = (krow,kcol) - 1 results in a spatial output shape equal to the
            spatial input shape. Otherwise, (prow,pcol)=(0,0) for padding="valid".
        '''
        if self.padding == "same":
            return tuple(np.array(kernel_shape) - np.array(strides_shape))
        elif self.padding == "valid":
            return (0,0)
        else:
            raise ValueError(f"'padding' is one of 'same' or 'valid'. Received {self.padding}.")

    def initialize_kernel_shape(self):
        self.kernel_shape = self.kernel.shape[:2]

    def initialize_spatial_input_shape(self):
        self.initialize_input_shape_params()
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

    def get_same_padding_spatial_output_shape(self,spatial_input_shape,strides):
        return np.floor((np.array(spatial_input_shape) - 1) / np.array(strides)) + 1

    def get_valid_padding_spatial_output_shape(self, spatial_input_shape, kernel_shape, strides):
        return np.floor((np.array(spatial_input_shape) - np.array(kernel_shape)) / np.array(strides)) + 1

    def initialize_same_padding_spatial_output_shape(self):
        if not hasattr(self,"spatial_input_shape"):
            self.initialize_spatial_input_shape()

        return self.get_same_padding_spatial_output_shape(self.spatial_input_shape,self.strides)

    def initialize_valid_padding_spatial_output_shape(self):
        if not hasattr(self,"spatial_input_shape"):
            self.initialize_spatial_input_shape()

        return self.get_valid_padding_spatial_output_shape(self.spatial_input_shape,self.kernel_shape,self.strides)

    def get_spatial_output_shape(self, spatial_input_shape, kernel_shape, strides):
        if self.padding == "same":
            spatial_output_shape = self.get_same_padding_spatial_output_shape(spatial_input_shape,strides)
        elif self.padding == "valid":
            spatial_output_shape = self.get_valid_padding_spatial_output_shape(spatial_input_shape,kernel_shape,strides)
        else:
            raise ValueError(f"'padding' is one of 'same' or 'valid'. Received {self.padding}.")

        spatial_output_shape = list(map(int,spatial_output_shape))
        return spatial_output_shape

    def initialize_spatial_output_shape(self):
        if self.padding == "same":
            spatial_output_shape = self.initialize_same_padding_spatial_output_shape()
        elif self.padding == "valid":
            spatial_output_shape = self.initialize_valid_padding_spatial_output_shape()
        else:
            raise ValueError(f"'padding' is one of 'same' or 'valid'. Received {self.padding}.")

        self.spatial_output_shape = list(map(int,spatial_output_shape))