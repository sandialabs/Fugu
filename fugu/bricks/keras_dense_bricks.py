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

    def __init__(self, units, weights=1.0, thresholds=0.5, name=None, prev_layer_prefix="pooling_", data_format="channels_last", input_shape=None, biases=None):
        super().__init__()
        self.is_built = False
        self.name = name
        self.supported_codings = ["binary-L"]
        self.weights = weights
        self.thresholds = thresholds
        self.data_format = data_format
        self.biases = biases
        # TODO: Output shape should be 1 dimension of shape (product(output_shape),)
        # TODO: Change 'output_shape' to 'output_units' to match tensorflow's Keras terminology.
        # TODO: 'output_units' should be a 1D array of shape (units,)
        self.output_units = units
        self.metadata = {'dense_output_units': self.output_units}
        self.prev_layer_prefix = prev_layer_prefix
        self.input_shape = input_shape

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
        if self.input_shape is None:
            self.input_shape = self.metadata["{}output_shape".format(self.prev_layer_prefix)]

        self.spatial_input_shape = self.get_spatial_input_shape()
        self.metadata["dense_input_shape"] = self.input_shape
        self.output_shape = (*np.array(self.input_shape)[:3],self.output_units)
        self.spatial_output_shape = self.get_spatial_output_shape()

        output_codings = [input_codings[0]]

        complete_node = self.name + "_complete"
        begin_node = self.name + "_begin"

        graph.add_node(begin_node, index=-1, threshold=0.0, decay=0.0, p=1.0, potential=0.0)
        graph.add_node(complete_node, index=0, threshold=0.9, decay=0.0, p=1.0, potential=0.0)

        graph.add_edge(control_nodes[0]["complete"], complete_node, weight=0.0, delay=1)
        graph.add_edge(control_nodes[0]["begin"], begin_node, weight=0.0, delay=1)

        self.check_thresholds_shape()
        self.check_weights_shape()

        # output neurons/nodes
        output_lists = [[]]
        for row in np.arange(0, self.spatial_output_shape[0]):
            for col in np.arange(0, self.spatial_output_shape[1]):
                for k in np.arange(self.output_units):
                    graph.add_node(f'{self.name}d{k}{row}{col}', index=(row,col,k), threshold=self.thresholds[0,row,col,k], decay=1.0, p=1.0, potential=0.0)
                    output_lists[0].append(f'{self.name}d{k}{row}{col}')

        # Biases for dense layer
        if self.biases is not None:
            self.check_biases_shape()
            # biases neurons/nodes; one node per kernel/channel in filter
            for k in np.arange(self.output_units):
                graph.add_node(f'{self.name}b{k}', index=(98,k), threshold=0.0, decay=1.0, p=1.0, potential=0.1)

            # Construct edges connecting biases node(s) to output nodes
            for k in np.arange(self.output_units):
                for row in np.arange(0, self.spatial_output_shape[0]):
                    for col in np.arange(0, self.spatial_output_shape[1]):
                        graph.add_edge(f'{self.name}b{k}',f'{self.name}d{k}{row}{col}', weight=self.biases[k], delay=1)
                        print(f"{self.name}b{k} ---> {self.name}d{k}{row}{col} : weight = {self.biases[k]}")

        self.connect_input_and_output_neurons_4(input_lists,graph)

        self.is_built = True
        return (graph, self.metadata, [{"complete": complete_node, "begin": begin_node}], output_lists, output_codings,)

    def connect_input_and_output_neurons_1(self,input_lists,graph):
        # Collect Inputs
        prev_layer = np.reshape(input_lists[0], self.input_shape)

        # Construct edges connecting input and output nodes
        for outrow in np.arange(self.spatial_output_shape[0]):  # loop over output neurons
            for outcol in np.arange(self.spatial_output_shape[1]): # loop over output neurons
                for outchan in np.arange(self.output_units):

                    for inrow in np.arange(self.spatial_input_shape[0]):  # loop over input neurons
                        for incol in np.arange(self.spatial_input_shape[1]):  # loop over input neurons
                            for inchan in np.arange(self.nChannelsInput): # loop over input neuron channels
                                graph.add_edge(prev_layer[0,inrow,incol,inchan], f'{self.name}d{outchan}{outrow}{outcol}', weight=self.weights[inchan,outchan], delay=1)
                                # print(f" p{inchan}{inrow}{incol} --> d{outchan}{outrow}{outcol}   weight: {self.weights[inchan,outchan]}")

    def connect_input_and_output_neurons_2(self,input_lists,graph):
        # Collect Inputs
        prev_layer = np.reshape(input_lists[0], self.input_shape)

        for inrow in np.arange(self.spatial_input_shape[0]):  # loop over input neurons
            for incol in np.arange(self.spatial_input_shape[1]):  # loop over input neurons
                for inchan in np.arange(self.nChannelsInput): # loop over input neuron channels

                    # Construct edges connecting input and output nodes
                    for outrow in np.arange(self.spatial_output_shape[0]):  # loop over output neurons
                        for outcol in np.arange(self.spatial_output_shape[1]): # loop over output neurons
                            for outchan in np.arange(self.output_units):
                                graph.add_edge(prev_layer[0,inrow,incol,inchan], f'{self.name}d{outchan}{outrow}{outcol}', weight=self.weights[inchan,outchan], delay=1)
                                # print(f" p{inchan}{inrow}{incol} --> d{outchan}{outrow}{outcol}   weight: {self.weights[inchan,outchan]}")

    def connect_input_and_output_neurons_3(self,input_lists,graph):
        # Collect Inputs
        prev_layer = np.reshape(input_lists[0], self.input_shape)

        for inrow in np.arange(self.spatial_input_shape[0]):  # loop over input neurons
            for incol in np.arange(self.spatial_input_shape[1]):  # loop over input neurons
                for inchan in np.arange(self.nChannelsInput): # loop over input neuron channels

                    # Construct edges connecting input and output nodes
                    for outrow in np.arange(self.spatial_output_shape[0]):  # loop over output neurons
                        for outcol in np.arange(self.spatial_output_shape[1]): # loop over output neurons
                            for outchan in np.arange(self.output_units):
                                graph.add_edge(prev_layer[0,inrow,incol,inchan], f'{self.name}d{outchan}{outrow}{outcol}', weight=self.weights[inchan,outchan], delay=1)
                                print(f" p{inchan}{inrow}{incol} --> d{outchan}{outrow}{outcol}   weight: {self.weights[inchan,outchan]}")

    def connect_input_and_output_neurons_4(self,input_lists,graph):
        # Collect Inputs
        prev_layer = np.reshape(input_lists[0], self.input_shape)

        for row in np.arange(self.spatial_input_shape[0]):  # loop over input neurons
            for col in np.arange(self.spatial_input_shape[1]):  # loop over input neurons
                for inchan in np.arange(self.nChannelsInput): # loop over input neuron channels

                    # Construct edges connecting input and output nodes
                    for outchan in np.arange(self.output_units):
                        graph.add_edge(prev_layer[0,row,col,inchan], f'{self.name}d{outchan}{row}{col}', weight=self.weights[inchan,outchan], delay=1)
                        print(f" p{inchan}{row}{col} --> d{outchan}{row}{col}   weight: {self.weights[inchan,outchan]}")

    def check_thresholds_shape(self):
        # Check for scalar value for thresholds or consistent thresholds shape
        expected_thresholds_shape = self.output_shape
        error_str = "Threshold shape {} does not equal the output neuron shape {}."
        self.thresholds = self.check_shape(self.thresholds,expected_thresholds_shape,error_str)

    def check_weights_shape(self):
        # Check for scalar value for weights or consistent weights shape
        # Weights is a matrix that that has output_units rows and flattened(input_shape) columns (excluding batch size).
        expected_weights_shape = (self.nChannelsInput,self.output_units)
        error_str = "Weights shape {} does not equal the necessary shape {}."
        self.weights = self.check_shape(self.weights, expected_weights_shape,error_str)

    def check_biases_shape(self):
        expected_biases_shape = (self.output_units,)
        error_str = "Biases shape {} does not equal the necessary shape {}."
        self.biases = self.check_shape(self.biases, expected_biases_shape,error_str)

    def check_shape(self, variable, expected_variable_shape, error_str):
        if not hasattr(variable, '__len__') and (not isinstance(variable, str)):
            variable = variable * np.ones(expected_variable_shape, dtype=float)
        else:
            if not type(variable) is np.ndarray:
                variable = np.array(variable)

            if variable.shape != expected_variable_shape:
                raise ValueError(error_str.format(variable.shape, expected_variable_shape))

        return variable

    def get_spatial_input_shape(self):
        self.batch_size, self.image_height, self.image_width, self.nChannelsInput = self.get_dense_input_shape_params()
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
        batch_size, image_height, image_width, self.nChannelsOutput = self.get_dense_output_shape_params()
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