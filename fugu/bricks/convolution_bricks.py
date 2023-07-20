#!/usr/bin/env python3

import logging
import numpy as np
from .bricks import Brick

class convolution_1d(Brick):
    'Convolution brick that assumes collapse_binary=False for all base-p values'
    """
    Convolution Function
    Michael Krygier
    mkrygie@sandia.gov
    
    """
    
    def __init__(self, pvector, filters, thresholds, basep, bits, name=None, mode='full'):
        super().__init__()
        self.is_built = False
        self.name = name
        self.supported_codings = ['binary-L']
        self.plength = len(pvector)
        self.filters = filters
        self.thresholds = thresholds
        self.basep = basep
        self.bits = bits
        self.mode = mode
        self.metadata = {'D': 2, 'basep': basep, 'bits': bits, 'convolution_mode': mode, 'convolution_input_shape': (self.plength,)}
        
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
        input_length = self.plength
        filter_length = len(self.filters)
        num_input_neurons = len(input_lists[0])
        
        # determine output neuron bounds based on the "mode"
        self.get_output_bounds()
        dim1 = self.bnds[1] - self.bnds[0] + 1
        self.metadata['convolution_output_shape'] = (dim1,)

        # Check for scalar value for thresholds
        if not hasattr(self.thresholds, '__len__') and (not isinstance(self.thresholds, str)):
            self.thresholds = self.thresholds * np.ones(dim1)
        else:
            if self.thresholds.shape != (dim1,):
                raise ValueError(f"Threshold length {self.thresholds.shape} does not equal the output neuron length ({dim1},)."
                )

        # output neurons/nodes
        output_lists = [[]]
        for i in np.arange(self.bnds[0], self.bnds[1] + 1):
            ix = i - self.bnds[0]
            graph.add_node(f'{self.name}g{i}', index=ix, threshold=self.thresholds[ix], decay=1.0, p=1.0, potential=0.0)
            output_lists[0].append(f'{self.name}g{i}')
        
        # Collect Inputs
        I = input_lists[0]
               
        # Construct edges connecting input and output nodes
        pwr = -1
        for i in np.arange(num_input_neurons):  # loop over input neurons
            coeff_i = np.mod(i, self.basep)
            if coeff_i == 0:
                pwr = pwr + 1
                if np.mod(pwr, self.bits) == 0:
                    pwr = 0
                continue

            row = np.unravel_index(i, (input_length, self.bits * self.basep))[0]
            for j in self.get_output_neurons(row, filter_length):  # loop over output neurons
                ix = j - row
                graph.add_edge(I[i], f'{self.name}g{j}', weight=coeff_i * self.basep**pwr * self.filters[ix], delay=1)
                logging.debug(f'coeff_i: {coeff_i}    power: {pwr}    input: {i}      output: {j}     filter id: {ix}     filter: {self.filters[ix]}')
                
        self.is_built=True        
        
        return (graph, self.metadata, [{'complete': complete_node, 'begin': begin_node}], output_lists, output_codings)

    def get_output_neurons(self, row, Bm):
        neuron_indices = []
        bnds = self.bnds

        if self.mode == "full":
            for i in np.arange(row, row + Bm):
                neuron_indices.append(i)

        if self.mode == "same" or self.mode == "valid":
            for i in np.arange(row, row + Bm):
                if (i >= bnds[0]) and (i <= bnds[1]):
                    neuron_indices.append(i)
                        
        return neuron_indices

    def get_output_bounds(self):
        Am = self.plength
        Bm = len(self.filters)
        Gm = Am + Bm - 1

        if self.mode == "full":
            lb = 0
            ub = Gm - 1
            self.bnds = np.array([lb, ub], dtype=int)
            
        if self.mode == "same":
            apos = Am 
            gpos = Gm 

            lb = np.floor(0.5 * (gpos - apos))
            ub = np.floor(0.5 * (gpos + apos) - 1)
            self.bnds = np.array([lb, ub], dtype=int)

        if self.mode == "valid":
            lmins = np.amin([Am, Bm])
            lb = lmins - 1
            ub = Gm - lmins
            self.bnds = np.array([lb, ub], dtype=int)

class convolution_2d(Brick):
    'Convolution brick that assumes collapse_binary=False for all base-p values'
    """
    Convolution Function
    Michael Krygier
    mkrygie@sandia.gov
    
    """
    
    def __init__(self, pvector, filters, thresholds, basep, bits, name=None, mode='full'):
        super().__init__()
        self.is_built = False
        self.name = name
        self.supported_codings = ['binary-L']
        self.pshape = np.array(pvector).shape
        self.filters = filters
        self.thresholds = thresholds
        self.basep = basep
        self.bits = bits
        self.mode = mode
        self.metadata = {'D': 2, 'basep': basep, 'bits': bits, 'convolution_mode': mode, 'convolution_input_shape': self.pshape}
        
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
        Bm, Bn = np.array(self.filters).shape

        # determine output neuron bounds based on the "mode"
        self.get_output_bounds()
        dim1, dim2 = self.bnds[1,0] - self.bnds[0,0] + 1, self.bnds[1,1] - self.bnds[0,1] + 1
        self.metadata['convolution_output_shape'] = (dim1,dim2)

        # Check for scalar value for thresholds
        if not hasattr(self.thresholds, '__len__') and (not isinstance(self.thresholds, str)):
            self.thresholds = self.thresholds * np.ones((dim1,dim2))
        else:
            if self.thresholds.shape != (dim1, dim2):
                raise ValueError(f"Threshold shape {self.thresholds.shape} does not equal the output neuron shape ({dim1},{dim2}).")

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
                logging.debug(f'{cnt}     coeff_i: {coeff_i}    power: {pwr}    input: {k}      output: {i}{j}     filter: {self.filters[ix][jx]}     I: {np.unravel_index(k,(Am,An,self.bits*self.basep))}')
                
        self.is_built=True
        
        return (graph, self.metadata, [{'complete': complete_node, 'begin': begin_node}], output_lists, output_codings)
    
    def get_output_neurons(self,row,col,Bm,Bn):
        neuron_indices = []
        bnds = self.bnds

        if self.mode == "full":
            for i in np.arange(row, row + Bm):
                for j in np.arange(col, col + Bn):
                    neuron_indices.append((i, j))

        if self.mode == "same" or self.mode == "valid":
            for i in np.arange(row, row + Bm):
                if (i >= bnds[0, 0]) and (i <= bnds[1, 0]):
                    for j in np.arange(col, col + Bn):
                        if (j >= bnds[0, 1]) and (j <= bnds[1, 1]):
                            neuron_indices.append((i, j))

        return neuron_indices

    def get_output_bounds(self):
        Am, An = self.pshape
        Bm, Bn = np.array(self.filters).shape
        Gm, Gn = Am + Bm - 1, An + Bn - 1

        if self.mode == "full":
            lb = [0, 0]
            ub = [Gm - 1, Gn - 1]
            self.bnds = np.array([lb, ub], dtype=int)

        if self.mode == "same":
            apos = np.array([Am, An])
            gpos = np.array([Gm, Gn])

            lb = np.floor(0.5 * (gpos - apos))
            ub = np.floor(0.5 * (gpos + apos) - 1)
            self.bnds = np.array([lb, ub], dtype=int)

        if self.mode == "valid":
            lmins = np.minimum((Am, An), (Bm, Bn))
            lb = lmins - 1
            ub = np.array([Gm, Gn]) - lmins
            self.bnds = np.array([lb, ub], dtype=int)
