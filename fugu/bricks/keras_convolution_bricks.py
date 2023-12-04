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
        self.get_output_bounds_alt()

        # Check for scalar value for thresholds
        if not hasattr(self.thresholds, '__len__') and (not isinstance(self.thresholds, str)):
            self.thresholds = self.thresholds * np.ones(self.output_shape)
        else:
            if self.thresholds.shape != self.output_shape:
                raise ValueError(f"Threshold shape {self.thresholds.shape} does not equal the output neuron shape {self.output_shape}.")

        # # output neurons/nodes
        # output_lists = [[]]
        # for i in np.arange(self.bnds[0,0],self.bnds[1,0] + 1):
        #     ix = i - self.bnds[0,0]
        #     for j in np.arange(self.bnds[0,1],self.bnds[1,1] + 1):
        #         jx = j - self.bnds[0,1]
        #         graph.add_node(f'{self.name}g{i}{j}', index=(ix,jx), threshold=self.thresholds[ix][jx], decay=1.0, p=1.0, potential=0.0)
        #         output_lists[0].append(f'{self.name}g{i}{j}')

        # # Biases for convolution
        # if self.biases is not None:
        #     # biases neurons/nodes; one node per kernel/channel in filter
        #     graph.add_node(f'{self.name}b', index=(99,0), threshold=-1.0, decay=1.0, p=1.0, potential=0.0)

        #     # Construct edges connecting biases node(s) to output nodes
        #     for i in np.arange(self.bnds[0,0],self.bnds[1,0] + 1):
        #         for j in np.arange(self.bnds[0,1],self.bnds[1,1] + 1):
        #             graph.add_edge(f'{self.name}b',f'{self.name}g{i}{j}', weight=self.biases, delay=1)

        # output neurons/nodes
        output_lists = self.create_output_neurons(graph)

        self.create_biases_nodes_and_synapses(graph)
        self.connect_input_and_output_neurons_alt(input_lists,graph)

        self.is_built=True

        return (graph, self.metadata, [{'complete': complete_node, 'begin': begin_node}], output_lists, output_codings)

    def create_output_neurons(self,graph):
        # output neurons/nodes
        output_lists = [[]]
        # TODO: Fixed non-continuous memory strides. This will result in poor performance.
        for ix, i in enumerate(np.arange(self.bnds[0,0],self.bnds[1,0] + 1,self.strides[0])):
            for jx, j in enumerate(np.arange(self.bnds[0,1],self.bnds[1,1] + 1,self.strides[1])):
                graph.add_node(f'{self.name}g{i}{j}', index=(ix,jx), threshold=self.thresholds[ix,jx], decay=1.0, p=1.0, potential=0.0)
                output_lists[0].append(f'{self.name}g{i}{j}')

        return output_lists

    def create_biases_nodes_and_synapses(self,graph):
        # Biases for convolution
        if self.biases is not None:
            # biases neurons/nodes; one node per kernel/channel in filter
            graph.add_node(f'{self.name}b', index=(99,0), threshold=-1.0, decay=1.0, p=1.0, potential=0.0)

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

        # Construct edges connecting input and output nodes
        pwr = -1
        cnt = -1
        for k in np.arange(num_input_neurons):  # loop over input neurons
            coeff_i = np.mod(k, self.basep)
            if coeff_i == 0:
                pwr = pwr + 1
                if np.mod(pwr, self.bits) == 0:
                    pwr = 0
                continue

            # loop over output neurons
            row, col = np.unravel_index(k, (Am, An, self.bits * self.basep))[0:2]
            for i, j in self.get_output_neurons(row, col, Bm, Bn):
                ix = i - row
                jx = j - col

                cnt += 1
                graph.add_edge(I[k], f'{self.name}g{i}{j}', weight=coeff_i * self.basep**pwr * self.filters[ix][jx], delay=1)
                logging.debug(f'{cnt}     coeff_i: {coeff_i}    power: {pwr}    input: {k}      output: {i}{j}     filter: {self.filters[ix][jx]}     I(row,col,Ck): {np.unravel_index(k,(Am,An,self.bits*self.basep))}     I[index]: {graph.nodes[I[k]]["index"]}')

    def connect_input_and_output_neurons_alt(self,input_lists,graph):
        # Get size/shape information from input arrays
        Am, An = self.pshape
        Bm, Bn = self.filters.shape

        num_input_neurons = len(input_lists[0])

        # Collect Inputs
        I = input_lists[0]

        output_neurons = {(row,col): self.get_output_neurons(row,col,Bm,Bn) for col in np.arange(An) for row in np.arange(Am)}
        output_neurons_alt = {(row,col): self.get_output_neurons_alt(row,col,Bm,Bn) for col in np.arange(An) for row in np.arange(Am)}

        # Construct edges connecting input and output nodes
        cnt = -1
        for k in np.arange(num_input_neurons):  # loop over input neurons
            # loop over output neurons
            row, col, pwr, Ck = np.unravel_index(k, (Am, An, self.bits, self.basep))
            if Ck == 0:
                continue

            for i, j in output_neurons_alt[(row,col)]:
                # ix = i - row
                # jx = j - col
                ix = i - row + (Bm - 1)
                jx = j - col + (Bn - 1)
                # ix = (Bm - 1) - i
                # jx = (Bn - 1) - j
                # need method to properly loop over kernel entries here

                cnt += 1
                graph.add_edge(I[k], f'{self.name}g{i}{j}', weight=Ck * self.basep**pwr * self.filters[ix][jx], delay=1)
                logging.debug(f'{cnt:3d}  A[m,n]: ({row:2d},{col:2d})   power: {pwr}    coeff_i: {Ck}    input: {k:3d}      output: {i}{j}   B[m,n]: ({ix:2d},{jx:2d})   filter: {self.filters[ix][jx]}     I(row,col,bit-pwr,basep-coeff): {np.unravel_index(k,(Am,An,self.bits,self.basep))}     I[index]: {graph.nodes[I[k]]["index"]}')

    def connect_input_and_output_neurons_alt2(self,input_lists,output_lists,graph):
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
        print("")
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
        pass

    def get_output_neurons(self,row,col,Bm,Bn):
        neuron_indices = []
        bnds = self.bnds
        Sm, Sn = self.strides

        #TODO: Do I need these conditions in this loop? Is it possible to simply use
        # bnds[0,0] to bnds[1,0] as the range for i and
        # bnds[0,1] to bnds[1,1] as the range for j here?
        for i in np.arange(row, row + Bm):
            if (i >= bnds[0, 0]) and (i <= bnds[1, 0]):
                for j in np.arange(col, col + Bn):
                    if (j >= bnds[0, 1]) and (j <= bnds[1, 1]):
                        neuron_indices.append((i, j))

        return neuron_indices

    def get_output_neurons_alt(self,row,col,Bm,Bn):
        neuron_indices = []
        Sm, Sn = self.strides

        for i in np.arange(row - Bm, row):
            if i+1 < 0 or np.mod(i+1, Sm) != 0 or i+1 > self.bnds[1,0]:
                continue
            for j in np.arange(col - Bn, col):
                if j+1 < 0 or np.mod(j+1, Sn) != 0 or j+1 > self.bnds[1,1]:
                    continue
                neuron_indices.append((i+1,j+1))

        # for i in np.arange(row, row - Bm, -1):
        #     if i < 0 or np.mod(i, Sm) != 0:
        #         continue
        #     for j in np.arange(col, col - Bn, -1):
        #         if j < 0 or np.mod(j, Sn) != 0:
        #             continue
        #         neuron_indices.append((i,j))
        return neuron_indices

    def get_output_bounds(self):
        input_shape = np.array(self.pshape)
        full_output_shape = np.array(self.pshape) + np.array(self.filters.shape) - 1
        mode_output_shape = np.array(self.output_shape)

        if self.mode == "same":
            ub = np.floor(0.5 * (mode_output_shape + input_shape) - 1)
            lb = ub - (mode_output_shape - 1)
            self.bnds = np.array([lb,ub],dtype=int) + 1
        elif self.mode == "valid":
            lmins = np.minimum(self.pshape,self.filters.shape)
            ub = full_output_shape - lmins
            lb = ub - (mode_output_shape - 1)
            self.bnds = np.array([lb,ub],dtype=int) - (np.array(self.strides) - 1)

    def get_output_bounds_alt(self):
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

    def __init__(self, input_shape, filters, thresholds, basep, bits, name=None, mode='same', strides=(1,1), biases=None):
        super().__init__()
        self.is_built = False
        self.name = name
        self.supported_codings = ['binary-L']
        self.pshape = input_shape # (batch size, height, width, nChannels), can assume batch size is 1 or None.
        self.filters = np.array(filters) # (kernel height, kernel width, nChannels, nFilters)
        self.thresholds = thresholds    # must match self.filters.shape
        self.basep = basep
        self.bits = bits
        self.mode = mode
        self.biases = biases   # shape must be (nFilters,)

        self.nChannels = self.pshape[-1]
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
        self.get_output_bounds_alt()

        # Check for scalar value for thresholds
        if not hasattr(self.thresholds, '__len__') and (not isinstance(self.thresholds, str)):
            self.thresholds = self.thresholds * np.ones(self.output_shape)
        else:
            if self.thresholds.shape != self.output_shape:
                raise ValueError(f"Threshold shape {self.thresholds.shape} does not equal the output neuron shape {self.output_shape}.")

        # output neurons/nodes
        output_lists = [[]]
        # TODO: Fixed non-continuous memory strides. This will result in poor performance.
        for kx in np.arange(self.nFilters):
            for ix, i in enumerate(np.arange(self.bnds[0,0],self.bnds[1,0] + 1,self.strides[0])):
                for jx, j in enumerate(np.arange(self.bnds[0,1],self.bnds[1,1] + 1,self.strides[1])):
                    graph.add_node(f'{self.name}g{i}{j}{kx}', index=(ix,jx,kx), threshold=self.thresholds[0,ix,jx,kx], decay=1.0, p=1.0, potential=0.0)
                    output_lists[0].append(f'{self.name}g{i}{j}{kx}')

        # Biases for convolution
        if self.biases is not None:
            # biases neurons/nodes; one node per kernel/channel in filter
            for k in np.arange(self.nFilters):
                graph.add_node(f'{self.name}b{k}', index=(99,k), threshold=-1.0, decay=1.0, p=1.0, potential=0.0)

            # Construct edges connecting biases node(s) to output nodes
            for i in np.arange(self.bnds[0,0],self.bnds[1,0] + 1,self.strides[0]):
                for j in np.arange(self.bnds[0,1],self.bnds[1,1] + 1,self.strides[1]):
                    for k in np.arange(self.nFilters):
                        graph.add_edge(f'{self.name}b{k}',f'{self.name}g{i}{j}{k}', weight=self.biases[k], delay=1)

        self.connect_input_and_output_neurons(input_lists,graph)

        # for filter in np.arange(self.nFilters):
        #     for channel in np.arange(self.nChannels):
        #         for i in np.arange(self.bnds[0,0],self.bnds[1,0] + 1,self.strides[0]):
        #             for j in np.arange(self.bnds[0,1],self.bnds[1,1] + 1,self.strides[1]):
        #                 for row, col in self.get_input_neurons(i, j, Bm, Bn):
        #                     ix = i - row
        #                     jx = j - col
        #                     k = np.ravel_multi_index((i,j,))

        self.is_built=True

        return (graph, self.metadata, [{'complete': complete_node, 'begin': begin_node}], output_lists, output_codings)

    def connect_input_and_output_neurons(self,input_lists,graph):
        # Get size/shape information from input arrays
        batch_size, Am, An, nChannels = self.pshape
        Bm, Bn = self.filters.shape[:2]

        I = np.array(input_lists[0])
        num_input_neurons_per_channel = Am * An * self.basep * self.bits
        num_input_neurons = len(input_lists[0])

        # I = np.array(input_lists[0]).reshape(-1,num_input_neurons_per_channel)
        # Construct edges connecting input and output nodes
        pwr = -1
        cnt = -1
        # for channel in np.arange(nChannels):
        for filter in np.arange(self.nFilters):
            for k in np.arange(num_input_neurons):  # loop over input neurons
                coeff_i = np.mod(k, self.basep)
                if coeff_i == 0:
                    pwr = pwr + 1
                    if np.mod(pwr, self.bits) == 0:
                        pwr = 0
                    continue

                # loop over output neurons
                row, col, channel = np.unravel_index(k, (Am, An, self.nChannels, self.bits * self.basep))[0:3]
                for i, j in self.get_output_neurons(row, col, Bm, Bn):
                    ix = i - row
                    jx = j - col

                    cnt += 1
                    graph.add_edge(I[k], f'{self.name}g{i}{j}{filter}', weight=coeff_i * self.basep**pwr * self.filters[ix,jx,channel,filter], delay=1)
                    print(f'{cnt}     coeff_i: {coeff_i}    power: {pwr}    input: {k}      output: {i}{j}{filter}     filter: {self.filters[ix,jx,channel,filter]}     I(row,col,Ck): {np.unravel_index(k,(Am,An,self.nChannels,self.bits*self.basep))}     I[index]: {graph.nodes[I[k]]["index"]}')

    def get_input_neurons(self,row,col,Bm,Bn):
        neuron_indices = []
        Am, An = self.pshape[1:3]

        for i in np.arange(row, row + Bm):
            if (i <= Am):
                for j in np.arange(col, col + Bn):
                    if (j <= An):
                        neuron_indices.append((i,j))

        return neuron_indices

    def get_output_neurons(self,row,col,Bm,Bn):
        neuron_indices = []
        bnds = self.bnds
        Sm, Sn = self.strides

        #TODO: Do I need these conditions in this loop? Is it possible to simply use
        # bnds[0,0] to bnds[1,0] as the range for i and
        # bnds[0,1] to bnds[1,1] as the range for j here?
        for i in np.arange(row, row + Bm):
            if (i >= bnds[0, 0]) and (i <= bnds[1, 0]):
                for j in np.arange(col, col + Bn):
                    if (j >= bnds[0, 1]) and (j <= bnds[1, 1]):
                        neuron_indices.append((i, j))

        return neuron_indices

    def get_output_neurons_alt(self,row,col,Bm,Bn):
        neuron_indices = []
        Sm, Sn = self.strides

        for i in np.arange(row - Bm, row):

            if i+1 < 0 or np.mod(i+1, Sm) != 0:
                continue
            for j in np.arange(col - Bn, col):
                if j+1 < 0 or np.mod(j+1, Sn) != 0:
                    continue
                neuron_indices.append((i+1,j+1))
        return neuron_indices

    def get_output_bounds(self):
        input_shape = np.array(self.pshape)[1:3]
        full_output_shape = np.array(self.pshape)[1:3] + np.array(self.filters.shape)[:2] - 1
        mode_output_shape = np.array(self.output_shape)[1:3]

        if self.mode == "same":
            ub = np.floor(0.5 * (mode_output_shape + input_shape) - 1)
            lb = ub - (mode_output_shape - 1)
            self.bnds = np.array([lb,ub],dtype=int) + 1
        elif self.mode == "valid":
            lmins = np.minimum(self.pshape[1:3],self.filters.shape[:2])
            ub = full_output_shape - lmins
            lb = ub - (mode_output_shape - 1)
            self.bnds = np.array([lb,ub],dtype=int) - (np.array(self.strides) - 1)

    def get_output_bounds_alt(self):
        input_shape = np.array(self.pshape)[1:3]
        kernel_shape = np.array(self.filters.shape)[:2]
        full_output_shape = input_shape + kernel_shape - 1
        mode_output_shape = np.array(self.output_shape)[1:3]

        if self.mode == "same":
            lb = np.floor(0.5 * (full_output_shape - input_shape))
            ub = np.floor(0.5 * (full_output_shape + input_shape) - 1)
            self.bnds = np.array([lb, ub], dtype=int)

        if self.mode == "valid":
            lmins = np.minimum(input_shape, kernel_shape)
            lb = lmins - 1
            ub = np.array(full_output_shape) - lmins
            self.bnds = np.array([lb, ub], dtype=int)

    def get_output_shape(self):
        strides_shape = np.array(self.strides)
        input_shape = np.array(self.pshape)[1:3]
        kernel_shape = np.array(self.filters.shape)[:2]
        nFilters = self.nFilters

        p = 0.5 if self.mode == "same" else 0
        output_shape = np.floor((input_shape + 2*p - kernel_shape)/strides_shape + 1).astype(int)
        self.output_shape = (1,) + tuple(output_shape) + (nFilters,)