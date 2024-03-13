# isort: skip_file
# fmt: off
import numpy as np
import pytest

from fugu.backends import snn_Backend
from fugu.bricks.dense_bricks import dense_layer_1d, dense_layer_2d
from fugu.bricks.input_bricks import BaseP_Input, Vector_Input
from fugu.bricks.keras_pooling_bricks import keras_pooling_2d_4dinput as pooling_2d
from fugu.scaffold import Scaffold
from fugu.utils.keras_helpers import keras_convolve2d, keras_convolve2d_4dinput, generate_keras_kernel, generate_mock_image, keras_convolution2d_output_shape, keras_convolution2d_output_shape_4dinput
from fugu.utils.whetstone_conversion import whetstone_2_fugu

import pandas as pd
from tensorflow import constant as tf_constant
from tensorflow.keras import Model, initializers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, BatchNormalization

from ..helpers import ConvolutionParams, PoolingParams, KerasParams

Spiking_BRelu = pytest.importorskip("whetstone.layers", reason=f"Whetstone package not installed. Skipping test file {__file__} because of module dependency.").Spiking_BRelu
layer_utils = pytest.importorskip("whetstone.utils", reason=f"Whetstone package not installed. Skipping test file {__file__} because of module dependency.").layer_utils

@pytest.mark.whetstone
class Test_Whetstone_2_Fugu_MaxPoolingLayer:
    def setup_method(self):
        self.basep = 4
        self.bits = 3

    @pytest.mark.parametrize("pool_size", [(1,2),(2,1),(2,2),(2,3),(3,2),(3,3)])
    @pytest.mark.parametrize("pool_strides", [(1,1),(1,2),(2,1),(2,2),(1,3),(3,1),(2,3),(3,2),(3,3)])
    @pytest.mark.parametrize("pool_padding", ["same", "valid"])
    @pytest.mark.parametrize("pool_method", ["max"])
    @pytest.mark.parametrize("nChannels,nFilters", [(2,3), (1,1)])
    def test_keras_max_pooling_layer(self, pool_size, pool_strides, pool_padding, pool_method, nChannels, nFilters):

        convo_obj = ConvolutionParams(image_height=5, image_width=5, nChannels=nChannels, kernel_height=2, kernel_width=2, nFilters=nFilters, biases=None)
        convo_obj.biases = convo_obj.get_random_biases_within_answer_range()
        convo_obj._set_convolution_answer_boolean()
        pool_obj = PoolingParams(convo_obj, pool_size=pool_size, pool_strides=pool_strides, pool_padding=pool_padding, pool_method=pool_method)

        keras_obj = KerasParams([convo_obj,pool_obj])
        keras_obj.pool_input = (keras_obj.features_extractor(convo_obj.mock_image)[0].numpy() > 0.5).astype(int)
        keras_obj.pool_answer = (keras_obj.features_extractor(convo_obj.mock_image)[1].numpy() > 0.5).astype(int)
        expected_spike_count = keras_obj.pool_answer.sum()

        result, graph = run_whetstone_to_fugu_utility(convo_obj.mock_image, self.basep, self.bits, keras_obj.model)
        calculated_spike_count = len(get_pooling_neurons_result_only(result,graph).index)
        assert expected_spike_count == calculated_spike_count

def get_neuron_numbers(name_prefix, graph):
    neuron_numbers = []
    for key in graph.nodes.keys():
        if key.startswith(name_prefix):
            neuron_numbers.append(graph.nodes[key]['neuron_number'])

    return np.array(neuron_numbers)

def get_pooling_neurons_result_only(result, graph):
    pool_neuron_numbers = get_neuron_numbers('pool_layer_',graph)
    sub_result = result[result['neuron_number'].isin(pool_neuron_numbers)]
    return sub_result

def run_whetstone_to_fugu_utility(mock_image, basep, bits, keras_model):
    scaffold = Scaffold()
    scaffold.add_brick(BaseP_Input(mock_image,p=basep,bits=bits,collapse_binary=False,name="I",time_dimension=False),"input")
    scaffold = whetstone_2_fugu(keras_model,basep,bits,scaffold=scaffold)
    graph = scaffold.lay_bricks()
    scaffold.summary(verbose=1)
    backend = snn_Backend()
    backend_args = {}
    backend.compile(scaffold, backend_args)
    result = backend.run(10)
    return result, graph

@pytest.mark.whetstone
class Test_Whetstone_2_Fugu_AveragePoolingLayer:
    def setup_method(self):
        self.basep = 4
        self.bits = 3

    @pytest.mark.parametrize("pool_size", [(1,2),(2,1),(2,2),(2,3),(3,2),(3,3)])
    @pytest.mark.parametrize("pool_strides", [(1,1),(1,2),(2,1),(2,2),(1,3),(3,1),(2,3),(3,2),(3,3)])
    @pytest.mark.parametrize("pool_padding", ["same", "valid"])
    @pytest.mark.parametrize("pool_method", ["average"])
    @pytest.mark.parametrize("nChannels,nFilters", [(2,3), (1,1)])
    def test_keras_average_pooling_layer(self, pool_size, pool_strides, pool_padding, pool_method, nChannels, nFilters):

        convo_obj = ConvolutionParams(image_height=5, image_width=5, nChannels=nChannels, kernel_height=2, kernel_width=2, nFilters=nFilters, biases=None)
        convo_obj.biases = convo_obj.get_random_biases_within_answer_range()
        convo_obj._set_convolution_answer_boolean()
        pool_obj = PoolingParams(convo_obj, pool_size=pool_size, pool_strides=pool_strides, pool_padding=pool_padding, pool_method=pool_method)

        keras_obj = KerasParams([convo_obj,pool_obj])
        keras_obj.pool_input = (keras_obj.features_extractor(convo_obj.mock_image)[0].numpy() > 0.5).astype(int)
        keras_obj.pool_answer = (keras_obj.features_extractor(convo_obj.mock_image)[1].numpy() > 0.5).astype(int)
        expected_spike_count = keras_obj.pool_answer.sum()

        result, graph = run_whetstone_to_fugu_utility(convo_obj.mock_image, self.basep, self.bits, keras_obj.model)
        calculated_spike_count = len(get_pooling_neurons_result_only(result,graph).index)
        assert expected_spike_count == calculated_spike_count
