#!/usr/bin/env python3

import logging
import numpy as np
from .bricks import Brick

class keras_convolution_2d(Brick):
    'Convolution brick that assumes collapse_binary=False for all base-p values'
    """
    Convolution Function
    Michael Krygier
    mkrygie@sandia.gov

    """

    def __init__(self, input_shape, filters, thresholds, basep, bits, name=None, mode='same', strides=(1,1)):
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

        # Get size/shape information from input arrays
        Am, An = self.pshape
        Bm, Bn = self.filters.shape

        # determine output neuron bounds based on the "mode"
        self.get_output_bounds()

        # Check for scalar value for thresholds
        if not hasattr(self.thresholds, '__len__') and (not isinstance(self.thresholds, str)):
            self.thresholds = self.thresholds * np.ones(self.output_shape)
        else:
            if self.thresholds.shape != self.output_shape:
                raise ValueError(f"Threshold shape {self.thresholds.shape} does not equal the output neuron shape {self.output_shape}.")

        num_input_neurons = len(input_lists[0])

        # output neurons/nodes
        output_lists = [[]]
        for i in np.arange(self.bnds[0,0],self.bnds[1,0] + 1):
            ix = i - self.bnds[0,0]
            for j in np.arange(self.bnds[0,1],self.bnds[1,1] + 1):
                jx = j - self.bnds[0,1]
                graph.add_node(f'{self.name}g{i}{j}', index=(ix,jx), threshold=self.thresholds[ix][jx], decay=1.0, p=1.0, potential=0.0)
                output_lists[0].append(f'{self.name}g{i}{j}')

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

        self.is_built=True

        return (graph, self.metadata, [{'complete': complete_node, 'begin': begin_node}], output_lists, output_codings)

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

    def get_output_bounds(self):
        input_shape = np.array(self.pshape)
        full_output_shape = np.array(self.pshape) + np.array(self.filters.shape) - 1
        mode_output_shape = np.array(self.output_shape)
        
        if self.mode == "same":
            ub = np.floor(0.5 * (full_output_shape + input_shape) - 1)
        elif self.mode == "valid":
            lmins = np.minimum(self.pshape,self.filters.shape)
            ub = full_output_shape - lmins

        lb = ub - (mode_output_shape - 1)
        self.bnds = np.array([lb,ub],dtype=int)

    def get_output_shape(self):
        strides_shape = np.array(self.strides)
        input_shape = np.array(self.pshape)
        kernel_shape = np.array(self.filters.shape)

        p = 0.5 if self.mode == "same" else 0
        output_shape = np.floor((input_shape + 2*p - kernel_shape)/strides_shape + 1).astype(int)
        self.output_shape = tuple(output_shape)

def input_index_to_matrix_entry(input_shape,basep,bits,index):
    Am, An = input_shape
    linearized_index = np.ravel_multi_index(index,tuple(np.repeat([1,2,basep],[1,2,2])))# zero-based linearized index

    return np.unravel_index(linearized_index,(Am,An,basep*bits))[:2]