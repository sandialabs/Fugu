#!/usr/bin/env python3

import logging
import numpy as np
from .bricks import Brick
from .metadata_utils import is_metadata_key_present, get_metadata_key_value

class dense_layer_1d(Brick):
    "Dense Layer brick"
    """
    Dense Layer
    Michael Krygier
    mkrygie@sandia.gov
    
    """

    def __init__(self, output_shape, weights, thresholds, name=None, layer_name="dense_1d"):
        super().__init__()
        self.is_built = False
        self.name = name
        self.supported_codings = ["binary-L"]
        self.weights = np.array(weights)
        self.thresholds = np.array(thresholds)
        self.metadata = {'isNeuralNetworkLayer': True, 'layer_name': layer_name, 'output_shape': output_shape}
        self.output_shape = output_shape

    def build(self, graph, metadata, control_nodes, input_lists, input_codings):
        """
        Build Pooling Layer brick.

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

        # if type(metadata) is list:
        #     self.metadata = {**metadata[0], **self.metadata}
        # else:
        #     self.metadata = {**metadata, **self.metadata}
        if is_metadata_key_present(metadata[0],'isNeuralNetworkLayer'):
            self.input_shape = get_metadata_key_value(metadata[0],'output_shape')

        assert hasattr(self, 'input_shape')
        self.metadata['input_shape'] = self.input_shape

        output_codings = [input_codings[0]]

        complete_node = self.name + "_complete"
        begin_node = self.name + "_begin"

        graph.add_node(begin_node, index=-1, threshold=0.0, decay=0.0, p=1.0, potential=0.0)
        graph.add_node(complete_node, index=0, threshold=0.9, decay=0.0, p=1.0, potential=0.0)

        graph.add_edge(control_nodes[0]["complete"], complete_node, weight=0.0, delay=1)
        graph.add_edge(control_nodes[0]["begin"], begin_node, weight=0.0, delay=1)

        num_input_neurons = len(input_lists[0])
        num_output_neurons = num_input_neurons
        
        output_size = np.prod(self.output_shape)
        input_size = np.prod(self.input_shape)

        # Check for scalar value for thresholds or consistent thresholds shape
        if not hasattr(self.thresholds, '__len__') and (not isinstance(self.thresholds, str)):
            self.thresholds = self.thresholds * np.ones(self.output_shape)
        else:
            if self.thresholds.shape != self.output_shape:
                raise ValueError(f"Threshold shape {self.thresholds.shape} does not equal the output neuron shape {self.output_shape}.")

        # Check for scalar value for weights or consistent weights shape
        if not hasattr(self.weights, '__len__') and (not isinstance(self.weights, str)):
            self.weights = self.weights * np.ones((output_size, input_size), dtype=float)
        else:
            if self.weights.shape != (output_size, input_size):
                raise ValueError(f"Weights shape {self.weights.shape} does not equal the necessary shape {(output_size, input_size)}.")
            
        # output neurons/nodes
        output_lists = [[]]
        for id in np.arange(num_output_neurons):
            graph.add_node(f'{self.name}d{id}', index=id, threshold=self.thresholds[id], decay=1.0, p=1.0, potential=0.0)
            output_lists[0].append(f'{self.name}d{id}')

        # Collect Inputs
        prev_layer = input_lists[0]

        # Construct edges connecting input and output nodes
        for i in np.arange(num_output_neurons):  # loop over output neurons
            for k in np.arange(num_input_neurons): # loop over input neurons
                graph.add_edge(prev_layer[k], f'{self.name}d{i}', weight=self.weights[i,k], delay=1)
                logging.debug(f" p{k} --> d{i}")

        self.is_built = True
        return (graph, self.metadata, [{"complete": complete_node, "begin": begin_node}], output_lists, output_codings,)


class dense_layer_2d(Brick):
    "Dense Layer brick"
    """
    Dense Layer
    Michael Krygier
    mkrygie@sandia.gov
    
    """

    def __init__(self, output_shape, weights=1.0, thresholds=0.9, name=None, layer_name="dense_2d"):
        super().__init__()
        self.is_built = False
        self.name = name
        self.supported_codings = ["binary-L"]
        self.weights = weights
        self.thresholds = thresholds
        self.metadata = {'isNeuralNetworkLayer': True, 'layer_name': layer_name, 'output_shape': output_shape}
        self.output_shape = output_shape

    def build(self, graph, metadata, control_nodes, input_lists, input_codings):
        """
        Build Dense Layer brick.

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
        # if type(metadata) is list:
        #     self.metadata = {**metadata[0], **self.metadata}
        # else:
        #     self.metadata = {**metadata, **self.metadata}

        if is_metadata_key_present(metadata[0],'isNeuralNetworkLayer'):
            self.input_shape = get_metadata_key_value(metadata[0],'output_shape')

        assert hasattr(self, 'input_shape')
        self.metadata['input_shape'] = self.input_shape

        output_codings = [input_codings[0]]

        complete_node = self.name + "_complete"
        begin_node = self.name + "_begin"

        graph.add_node(begin_node, index=-1, threshold=0.0, decay=0.0, p=1.0, potential=0.0)
        graph.add_node(complete_node, index=0, threshold=0.9, decay=0.0, p=1.0, potential=0.0)

        graph.add_edge(control_nodes[0]["complete"], complete_node, weight=0.0, delay=1)
        graph.add_edge(control_nodes[0]["begin"], begin_node, weight=0.0, delay=1)

        output_size = np.prod(self.output_shape)
        input_size = np.prod(self.input_shape)

        # Check for scalar value for thresholds or consistent thresholds shape
        if not hasattr(self.thresholds, '__len__') and (not isinstance(self.thresholds, str)):
            self.thresholds = self.thresholds * np.ones(self.output_shape)
        else:
            if not type(self.thresholds) is np.ndarray:
                self.thresholds = np.array(self.thresholds)

            if self.thresholds.shape != self.output_shape:
                raise ValueError(f"Threshold shape {self.thresholds.shape} does not equal the output neuron shape {self.output_shape}.")

        # Check for scalar value for weights or consistent weights shape
        if not hasattr(self.weights, '__len__') and (not isinstance(self.weights, str)):
            self.weights = self.weights * np.ones((output_size, input_size), dtype=float)
        else:
            if not type(self.weights) is np.ndarray:
                self.weights = np.array(self.weights)

            if self.weights.shape != (output_size, input_size):
                raise ValueError(f"Weights shape {self.weights.shape} does not equal the necessary shape {(output_size, input_size)}.")
            
        # output neurons/nodes
        output_lists = [[]]
        for row in np.arange(self.output_shape[0]):
            for col in np.arange(self.output_shape[1]):
                graph.add_node(f'{self.name}d{row}_{col}', index=(row,col), threshold=self.thresholds[row,col], decay=1.0, p=1.0, potential=0.0)
                output_lists[0].append(f'{self.name}d{row}_{col}')

        # Collect Inputs
        prev_layer = np.reshape(input_lists[0], self.input_shape)

        # Construct edges connecting input and output nodes
        wrow = -1
        for outrow in np.arange(self.output_shape[0]):  # loop over output neurons
            for outcol in np.arange(self.output_shape[1]): # loop over input neurons
                wcol = 0
                wrow = wrow + 1
                for inrow in np.arange(self.input_shape[0]):  # loop over input neurons
                    for incol in np.arange(self.input_shape[1]):  # loop over input neurons
                        graph.add_edge(prev_layer[inrow,incol], f'{self.name}d{outrow}_{outcol}', weight=self.weights[wrow,wcol], delay=1)
                        logging.debug(f" p{inrow}{incol} --> d{outrow}{outcol}   weight: {self.weights[wrow,wcol]}")
                        wcol = wcol + 1

        self.is_built = True
        return (graph, self.metadata, [{"complete": complete_node, "begin": begin_node}], output_lists, output_codings,)
