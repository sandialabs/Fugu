#!/usr/bin/env python3
# isort: skip_file
# fmt: off
import logging
import numpy as np
from .bricks import Brick

class keras_dense_2d_4dinput(Brick):
    "Dense Layer brick"
    """
    Dense Layer
    Michael Krygier
    mkrygie@sandia.gov
    
    """

    def __init__(self, output_shape, weights=1.0, thresholds=0.9, name=None, prev_layer_prefix="pooling_", data_format="channels_last"):
        super().__init__()
        self.is_built = False
        self.name = name
        self.supported_codings = ["binary-L"]
        self.weights = weights
        self.thresholds = thresholds
        self.data_format = data_format
        self.metadata = {'dense_output_shape': output_shape}
        self.prev_layer_prefix = prev_layer_prefix
        self.output_shape = output_shape
        self.spatial_output_shape = self.get_spatial_output_shape()

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
        if type(metadata) is list:
            self.metadata = {**metadata[0], **self.metadata}
        else:
            self.metadata = {**metadata, **self.metadata}

        #TODO: Add raise exception if dictionary key is absent from metadata (KeyError)
        self.input_shape = self.metadata["{}output_shape".format(self.prev_layer_prefix)]
        self.spatial_input_shape = self.get_spatial_input_shape()
        self.metadata["dense_input_shape"] = self.input_shape

        output_codings = [input_codings[0]]

        complete_node = self.name + "_complete"
        begin_node = self.name + "_begin"

        graph.add_node(begin_node, index=-1, threshold=0.0, decay=0.0, p=1.0, potential=0.0)
        graph.add_node(complete_node, index=0, threshold=0.9, decay=0.0, p=1.0, potential=0.0)

        graph.add_edge(control_nodes[0]["complete"], complete_node, weight=0.0, delay=1)
        graph.add_edge(control_nodes[0]["begin"], begin_node, weight=0.0, delay=1)

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
            self.weights = self.weights * np.ones((*self.spatial_output_shape, *self.spatial_input_shape, self.nChannels), dtype=float)
        else:
            if not type(self.weights) is np.ndarray:
                self.weights = np.array(self.weights)

            if self.weights.shape != (*self.spatial_output_shape, *self.spatial_input_shape, self.nChannels):
                raise ValueError(f"Weights shape {self.weights.shape} does not equal the necessary shape {(*self.spatial_output_shape, *self.spatial_input_shape, self.nChannels)}.")

        # output neurons/nodes
        output_lists = [[]]
        for row in np.arange(0, self.spatial_output_shape[0]):
            for col in np.arange(0, self.spatial_output_shape[1]):
                for channel in np.arange(self.nChannels):
                    graph.add_node(f'{self.name}d{channel}{row}{col}', index=(row,col,channel), threshold=self.thresholds[0,row,col,channel], decay=1.0, p=1.0, potential=0.0)
                    output_lists[0].append(f'{self.name}d{channel}{row}{col}')

        # Collect Inputs
        prev_layer = np.reshape(input_lists[0], self.input_shape)

        # Construct edges connecting input and output nodes
        for outrow in np.arange(self.spatial_output_shape[0]):  # loop over output neurons
            for outcol in np.arange(self.spatial_output_shape[1]): # loop over output neurons

                for inrow in np.arange(self.spatial_input_shape[0]):  # loop over input neurons
                    for incol in np.arange(self.spatial_input_shape[1]):  # loop over input neurons
                        for channel in np.arange(self.nChannels):
                            graph.add_edge(prev_layer[0,inrow,incol,channel], f'{self.name}d{channel}{outrow}{outcol}', weight=self.weights[outrow,outcol,inrow,incol,channel], delay=1)
                            print(f" p{channel}{inrow}{incol} --> d{channel}{outrow}{outcol}   weight: {self.weights[outrow,outcol,inrow,incol,channel]}")

        self.is_built = True
        return (graph, self.metadata, [{"complete": complete_node, "begin": begin_node}], output_lists, output_codings,)

    def get_spatial_input_shape(self):
        self.batch_size, self.image_height, self.image_width, self.nChannels = self.get_dense_input_shape_params()
        spatial_input_shape = (self.image_height, self.image_width)
        return spatial_input_shape
    
    def get_dense_input_shape_params(self):
        if self.data_format.lower() == "channels_last":
            batch_size, image_height, image_width, nChannels = self.input_shape
        elif self.data_format.lower() == "channels_first":
            batch_size, nChannels, image_height, image_width = self.input_shape
        else:
            raise ValueError(f"'data_format' is either 'channels_first' or 'channels_last'. Received {self.data_format}")

        return batch_size, image_height, image_width, nChannels  
    
    def get_spatial_output_shape(self):
        batch_size, image_height, image_width, nChannels = self.get_dense_output_shape_params()
        spatial_output_shape = (image_height, image_width)
        return spatial_output_shape
    
    def get_dense_output_shape_params(self):
        if self.data_format.lower() == "channels_last":
            batch_size, image_height, image_width, nChannels = self.output_shape
        elif self.data_format.lower() == "channels_first":
            batch_size, nChannels, image_height, image_width = self.output_shape
        else:
            raise ValueError(f"'data_format' is either 'channels_first' or 'channels_last'. Received {self.data_format}")

        return batch_size, image_height, image_width, nChannels    