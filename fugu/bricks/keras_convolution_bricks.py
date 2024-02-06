#!/usr/bin/env python3
# isort: skip_file
import logging
import numpy as np
from .bricks import Brick

# Turn off black formatting for this file
# fmt: off

class keras_convolution_2d(Brick):
    'Convolution brick that assumes collapse_binary=False for all base-p values'
    """
    Convolution Function
    Michael Krygier
    mkrygie@sandia.gov

    """

    def __init__(self, input_shape, filters, thresholds, basep, bits, name=None, mode='same', strides=(1,1), biases=None):
        super().__init__()
        self.is_built = False
        self.name = name
        self.supported_codings = ['binary-L']
        self.pshape = input_shape
        self.filters = np.array(filters)
        self.thresholds = thresholds
        self.basep = basep
        self.bits = bits
        self.mode = mode
        self.biases = biases

        if isinstance(strides,(list,tuple)):
            if len(strides) > 2:
                raise ValueError("Strides must be an integer or tuple/list of 2 integers.")
            else:
                strides = tuple(map(int,strides))
        elif isinstance(strides,(float,int)):
            strides = tuple(map(int,[strides,strides]))
        else:
            raise ValueError("Check strides input variable.")
        self.strides = strides
        self.get_output_shape()
        self.metadata = {'D': 2, 'basep': basep, 'bits': bits, 'convolution_mode': mode, 'convolution_input_shape': self.pshape, 'convolution_strides': self.strides, 'convolution_output_shape': self.output_shape}

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

        new_complete_node_name = self.name + '_complete'
        new_begin_node_name = self.name + '_begin'

        graph.add_node(new_begin_node_name   , index = -2, threshold = 0.0, decay =0.0, p=1.0, potential=0.0)
        graph.add_node(new_complete_node_name, index = -1, threshold = 0.9, decay =0.0, p=1.0, potential=0.0)


        graph.add_edge(control_nodes[0]['complete'], new_complete_node_name, weight=1.0, delay=1)
        graph.add_edge(control_nodes[0]['begin']   , new_begin_node_name   , weight=1.0, delay=1)

        complete_node = new_complete_node_name
        begin_node = new_begin_node_name

        # determine output neuron bounds based on the "mode"
        self.get_output_bounds()

        self.check_thresholds_shape()
        output_lists = self.create_output_neurons(graph)
        self.create_biases_nodes_and_synapses(graph)
        self.connect_input_and_output_neurons(input_lists,graph)

        self.is_built=True

        return (graph, self.metadata, [{'complete': complete_node, 'begin': begin_node}], output_lists, output_codings)

    def check_thresholds_shape(self):
        # Check for scalar value for thresholds
        if not hasattr(self.thresholds, '__len__') and (not isinstance(self.thresholds, str)):
            self.thresholds = self.thresholds * np.ones(self.output_shape)
        else:
            if self.thresholds.shape != self.output_shape:
                raise ValueError(f"Threshold shape {self.thresholds.shape} does not equal the output neuron shape {self.output_shape}.")

    def create_output_neurons(self,graph):
        # output neurons/nodes
        output_lists = [[]]
        for ix, i in enumerate(np.arange(self.bnds[0,0],self.bnds[1,0] + 1,self.strides[0])):
            for jx, j in enumerate(np.arange(self.bnds[0,1],self.bnds[1,1] + 1,self.strides[1])):
                graph.add_node(f'{self.name}g{i}{j}', index=(ix,jx), threshold=self.thresholds[ix,jx], decay=1.0, p=1.0, potential=0.0)
                output_lists[0].append(f'{self.name}g{i}{j}')

        return output_lists

    def create_biases_nodes_and_synapses(self,graph):
        # Biases for convolution
        if self.biases is not None:
            # biases neurons/nodes; one node per kernel/channel in filter
            graph.add_node(f'{self.name}b', index=(99,0), threshold=0.0, decay=1.0, p=1.0, potential=0.1)

            # Construct edges connecting biases node(s) to output nodes
            for i in np.arange(self.bnds[0,0],self.bnds[1,0] + 1,self.strides[0]):
                for j in np.arange(self.bnds[0,1],self.bnds[1,1] + 1,self.strides[1]):
                    graph.add_edge(f'{self.name}b',f'{self.name}g{i}{j}', weight=self.biases, delay=1)

    def connect_input_and_output_neurons(self,input_lists,graph):
        # Get size/shape information from input arrays
        Am, An = self.pshape
        Bm, Bn = self.filters.shape

        num_input_neurons = len(input_lists[0])

        # Collect Inputs
        I = input_lists[0]

        output_neurons = {(row,col): self.get_output_neurons(row,col,Bm,Bn) for col in np.arange(An) for row in np.arange(Am)}

        # Construct edges connecting input and output nodes
        cnt = -1
        for k in np.arange(num_input_neurons):  # loop over input neurons
            # loop over output neurons
            row, col, pwr, Ck = np.unravel_index(k, (Am, An, self.bits, self.basep))
            if Ck == 0:
                continue

            for i, j in output_neurons[(row,col)]:
                ix = i - row + (Bm - 1)
                jx = j - col + (Bn - 1)

                cnt += 1
                graph.add_edge(I[k], f'{self.name}g{i}{j}', weight=Ck * self.basep**pwr * self.filters[ix][jx], delay=1)
                logging.debug(f'{cnt:3d}  A[m,n]: ({row:2d},{col:2d})   power: {pwr}    coeff_i: {Ck}    input: {k:3d}      output: {i}{j}   B[m,n]: ({ix:2d},{jx:2d})   filter: {self.filters[ix][jx]}     I(row,col,bit-pwr,basep-coeff): {np.unravel_index(k,(Am,An,self.bits,self.basep))}     I[index]: {graph.nodes[I[k]]["index"]}')

    def connect_input_and_output_neurons_alt2(self,input_lists,output_lists,graph):
        '''
            Construct the input/output synapses (edges) by looping over the output neurons (outer loop) and assigning their edges to the input neurons (inner loop).

            This method is INCOMPLETE and may be removed later.
        '''
        # Get size/shape information from input arrays
        Am, An = self.pshape
        Bm, Bn = self.filters.shape
        Gm, Gn = self.bnds.shape

        num_input_neurons = len(input_lists[0])
        num_output_neurons = len(output_lists[0])

        # Collect Inputs
        I = input_lists[0]

        # Construct edges connecting input and output nodes
        cnt = -1
        # loop over output neurons
        for k in np.arange(num_output_neurons):
            row, col = np.unravel_index(k, (Gm, Gn))
            # loop over input neurons.
            for j in np.arange(num_input_neurons):
                aa, bb, pwr, Ck = np.unravel_index(j, (Am, An, self.bits, self.basep))
                ix = aa
                jx = bb
                graph.add_edge(I[j], f'{self.name}g{row}{col}', weight=Ck * self.basep**pwr * self.filters[ix][jx], delay=1)
        pass

    def get_output_neurons(self,row,col,Bm,Bn):
        neuron_indices = []
        Sm, Sn = self.strides

        for i in np.arange(row - Bm + 1, row + 1):
            if i < 0 or np.mod(i, Sm) != 0 or i > self.bnds[1,0]:
                continue
            for j in np.arange(col - Bn + 1, col + 1):
                if j < 0 or np.mod(j, Sn) != 0 or j > self.bnds[1,1]:
                    continue
                neuron_indices.append((i,j))

        return neuron_indices

    def get_output_bounds(self):
        input_shape = np.array(self.pshape)
        kernel_shape = np.array(self.filters.shape)
        full_output_shape = input_shape + kernel_shape - 1
        mode_output_shape = np.array(self.output_shape)

        if self.mode == "same":
            lb = np.floor(0.5 * (full_output_shape - input_shape))
            ub = np.floor(0.5 * (full_output_shape + input_shape) - 1)
            self.bnds = np.array([lb, ub], dtype=int)

        if self.mode == "valid":
            lmins = np.minimum(input_shape, kernel_shape)
            lb = lmins - 1
            ub = np.array(full_output_shape) - lmins
            self.bnds = np.array([lb, ub], dtype=int) - 1

    def get_output_shape(self):
        strides_shape = np.array(self.strides)
        input_shape = np.array(self.pshape)
        kernel_shape = np.array(self.filters.shape)

        p = 0.5 if self.mode == "same" else 0
        output_shape = np.floor((input_shape + 2*p - kernel_shape)/strides_shape + 1).astype(int)
        self.output_shape = tuple(output_shape)

def debug_input_index(input_lists, Am, An, basep, bits):
    for k in np.arange(len(input_lists[0])):
        row, col, pwr, Ck = np.unravel_index(k, (Am, An, bits, basep))
        print(f"{k:3d}  {np.mod(k,basep*bits):2d}  ({row},{col}) {Ck:2d}  {pwr:2d}")

def input_index_to_matrix_entry(input_shape,basep,bits,index):
    Am, An = input_shape
    linearized_index = np.ravel_multi_index(index,tuple(np.repeat([1,2,basep],[1,2,2])))# zero-based linearized index

    return np.unravel_index(linearized_index,(Am,An,basep*bits))[:2]




class keras_convolution_2d_4dinput(Brick):
    'Convolution brick that assumes collapse_binary=False for all base-p values'
    """
    Convolution Function
    Michael Krygier
    mkrygie@sandia.gov

    """

    def __init__(self, input_shape, filters, thresholds, basep, bits, name=None, mode='same', strides=(1,1), biases=None, data_format="channels_last"):
        # TODO: Add capability to handle Keras "data_format='channels_first'"
        # TODO: Rename 'mode' to 'padding' throughout brick.
        super().__init__()
        self.is_built = False
        self.name = name
        self.supported_codings = ['binary-L']
        self.input_shape = input_shape # (batch size, height, width, nChannels), can assume batch size is 1 or None.
        self.filters = np.array(filters) # (kernel height, kernel width, nChannels, nFilters)
        self.thresholds = thresholds    # must match self.filters.shape
        self.basep = basep
        self.bits = bits
        self.mode = mode
        self.biases = biases   # shape must be (nFilters,)

        self.nChannels = self.input_shape[-1]
        self.nFilters = self.filters.shape[-1]

        if isinstance(strides,(list,tuple)):
            if len(strides) > 2:
                raise ValueError("Strides must be an integer or tuple/list of 2 integers.")
            else:
                strides = tuple(map(int,strides))
        elif isinstance(strides,(float,int)):
            strides = tuple(map(int,[strides,strides]))
        else:
            raise ValueError("Check strides input variable.")
        self.strides = strides
        self.data_format = data_format
        self.get_spatial_input_shape()
        self.get_kernel_shape()
        self.get_output_shape()
        self.metadata = {'D': 2, 'basep': basep, 'bits': bits, 'convolution_mode': mode, 'convolution_input_shape': self.input_shape, 'convolution_strides': self.strides, 'convolution_output_shape': self.output_shape}

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

        # determine output neuron bounds based on the "mode"
        self.get_output_bounds()

        self.check_thresholds_shape()
        output_lists = self.create_output_neurons(graph)
        self.create_biases_nodes_and_synapses(graph,control_nodes)
        self.connect_input_and_output_neurons(input_lists,graph)

        self.is_built=True

        return (graph, self.metadata, [{'complete': complete_node, 'begin': begin_node}], output_lists, output_codings)

    def check_thresholds_shape(self):
        # Check for scalar value for thresholds
        if not hasattr(self.thresholds, '__len__') and (not isinstance(self.thresholds, str)):
            self.thresholds = self.thresholds * np.ones(self.output_shape)
        else:
            if self.thresholds.shape != self.output_shape:
                raise ValueError(f"Threshold shape {self.thresholds.shape} does not equal the output neuron shape {self.output_shape}.")

    def create_output_neurons(self, graph):
        # output neurons/nodes
        output_lists = [[]]
        for ix, i in enumerate(np.arange(self.bnds[0,0],self.bnds[1,0] + 1,self.strides[0])):
            for jx, j in enumerate(np.arange(self.bnds[0,1],self.bnds[1,1] + 1,self.strides[1])):
                for kx in np.arange(self.nFilters):
                    graph.add_node(f'{self.name}g{kx}{i}{j}', index=(ix,jx,kx), threshold=self.thresholds[0,ix,jx,kx], decay=1.0, p=1.0, potential=0.0)
                    output_lists[0].append(f'{self.name}g{kx}{i}{j}')

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
                for j in np.arange(self.bnds[0,0],self.bnds[1,0] + 1,self.strides[0]):
                    for i in np.arange(self.bnds[0,1],self.bnds[1,1] + 1,self.strides[1]):
                        graph.add_edge(f'{self.name}b{k}',f'{self.name}g{k}{j}{i}', weight=self.biases[k], delay=1)

    def connect_input_and_output_neurons(self,input_lists,graph):
        # Get size/shape information from input arrays
        batch_size, Am, An, nChannels = self.input_shape
        Bm, Bn = self.filters.shape[:2]

        I = np.array(input_lists[0])
        num_input_neurons = len(input_lists[0])

        input_neurons_2_output_neurons = {(row,col): self.get_output_neurons(row,col,Bm,Bn) for col in np.arange(An) for row in np.arange(Am)}

        # Construct edges connecting input and output nodes
        cnt = -1
        for k in np.arange(num_input_neurons):  # loop over input neurons

            # loop over output neurons
            row, col, channel, pwr, Ck = np.unravel_index(k, (Am, An, self.nChannels, self.bits,  self.basep))
            if Ck == 0:
                continue

            for i, j in input_neurons_2_output_neurons[(row,col)]:
                ix = i - row + (Bm - 1)
                jx = j - col + (Bn - 1)

                for filter in np.arange(self.nFilters):
                    cnt += 1
                    graph.add_edge(I[k], f'{self.name}g{filter}{i}{j}', weight=Ck * self.basep**pwr * self.filters[ix,jx,channel,filter], delay=2)
                    logging.debug(f'{cnt:3d}  A[m,n]: ({row:2d},{col:2d})   power: {pwr}    coeff_i: {Ck}    input: {k:3d}      output: {filter}{i}{j}   B[m,n]: ({ix:2d},{jx:2d})   filter: {self.filters[ix,jx,channel,filter]}     I(row,col,channel,bit-pwr,basep-coeff): {np.unravel_index(k,(Am,An,self.nChannels,self.bits,self.basep))}     I[index]: {graph.nodes[I[k]]["index"]}')

    def get_output_neurons(self,row,col,Bm,Bn):
        neuron_indices = []
        Sm, Sn = self.strides

        for i in np.arange(row - Bm + 1, row + 1):
            if i < 0 or np.mod(i, Sm) != 0 or i > self.bnds[1,0]:
                continue
            for j in np.arange(col - Bn + 1, col + 1):
                if j < 0 or np.mod(j, Sn) != 0 or j > self.bnds[1,1]:
                    continue
                neuron_indices.append((i,j))

        return neuron_indices

    def get_output_bounds(self):
        spatial_input_shape = np.array(self.spatial_input_shape)
        kernel_shape = np.array(self.kernel_shape)
        full_output_shape = spatial_input_shape + kernel_shape - 1

        if self.mode == "same":
            lb = np.floor(0.5 * (full_output_shape - spatial_input_shape))
            ub = np.floor(0.5 * (full_output_shape + spatial_input_shape) - 1)
            self.bnds = np.array([lb, ub], dtype=int)

        if self.mode == "valid":
            lmins = np.minimum(spatial_input_shape, kernel_shape)
            lb = lmins - 1
            ub = np.array(full_output_shape) - lmins
            self.bnds = np.array([lb, ub], dtype=int) - 1

    def get_output_shape(self):
        self.spatial_output_shape = self.get_spatial_output_shape()
        strides_shape = np.array(self.strides)
        spatial_input_shape = np.array(self.spatial_input_shape)
        kernel_shape = np.array(self.kernel_shape)
        nFilters = self.nFilters

        # p = 0.5 if self.mode == "same" else 0
        padding_shape = np.array(self.get_padded_zeros_count()) if self.mode == "same" else np.array((0,0))
        spatial_output_shape = np.floor((spatial_input_shape + padding_shape - kernel_shape)/strides_shape + 1).astype(int)
        self.output_shape = (1,) + tuple(spatial_output_shape) + (nFilters,)

    def get_padded_zeros_count(self):
        '''
            padding = strides*(output - 1) + kernel - input
        '''
        spatial_input_shape = np.array(self.spatial_input_shape)
        kernel_shape = np.array(self.kernel_shape)
        strides_shape = np.array(self.strides)
        spatial_output_shape = np.array(self.spatial_output_shape)

        padding_count = (strides_shape*(spatial_output_shape - 1) + kernel_shape - spatial_input_shape)
        padding_count[padding_count < 0] = 0.
        return tuple(padding_count.astype(int))

    def get_kernel_shape(self):
        self.kernel_shape = self.filters.shape[:2]

    def get_spatial_input_shape(self):
        self.batch_size, self.image_height, self.image_width, self.nChannels = self.get_input_shape_params()
        spatial_input_shape = (self.image_height, self.image_width)
        return spatial_input_shape

    def get_input_shape_params(self):
        if self.data_format.lower() == "channels_last":
            batch_size, image_height, image_width, nChannels = self.input_shape
        elif self.data_format.lower() == "channels_first":
            batch_size, nChannels, image_height, image_width = self.input_shape
        else:
            raise ValueError(f"'data_format' is either 'channels_first' or 'channels_last'. Received {self.data_format}")

        return batch_size, image_height, image_width, nChannels

    def same_padding_spatial_output_shape(self):
        if not hasattr(self,"spatial_input_shape"):
            self.spatial_input_shape = self.get_spatial_input_shape()

        return np.floor((np.array(self.spatial_input_shape) - 1) / np.array(self.strides)) + 1

    def valid_padding_spatial_output_shape(self):
        if not hasattr(self,"spatial_input_shape"):
            self.spatial_input_shape = self.get_spatial_input_shape()

        return np.floor((np.array(self.spatial_input_shape) - np.array(self.kernel_shape)) / np.array(self.strides)) + 1

    def get_spatial_output_shape(self):
        if self.mode.lower() == "same":
            spatial_output_shape = self.same_padding_spatial_output_shape()
        elif self.mode.lower() == "valid":
            spatial_output_shape = self.valid_padding_spatial_output_shape()
        else:
            raise ValueError(f"'padding' is one of 'same' or 'valid'. Received {self.mode}.")

        spatial_output_shape = list(map(int,spatial_output_shape))
        return spatial_output_shape