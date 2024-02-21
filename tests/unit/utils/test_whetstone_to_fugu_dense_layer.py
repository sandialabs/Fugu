# isort: skip_file
# fmt: off
import numpy as np
import pytest

from fugu.backends import snn_Backend
from fugu.bricks.keras_convolution_bricks import keras_convolution_2d as convolution_2d
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

from ..helpers import ConvolutionParams, PoolingParams, DenseParams, KerasParams, IntegerSequence, ArraySequence

Spiking_BRelu = pytest.importorskip("whetstone.layers", reason=f"Whetstone package not installed. Skipping test file {__file__} because of module dependency.").Spiking_BRelu
layer_utils = pytest.importorskip("whetstone.utils", reason=f"Whetstone package not installed. Skipping test file {__file__} because of module dependency.").layer_utils

@pytest.mark.whetstone
class Test_Whetstone_2_Fugu_DenseLayer:
    def setup_method(self):
        self.basep = 4
        self.bits = 3

    @pytest.mark.parametrize("units",np.arange(10))
    def test_utility_keras_dense_layer(self,units):
        convo_obj = ConvolutionParams(image_height=5, image_width=5, nChannels=2, kernel_height=2, kernel_width=2, nFilters=3, biases=None)
        convo_obj.biases = convo_obj.get_random_biases_within_answer_range()
        convo_obj._set_convolution_answer_boolean()
        pool_obj = PoolingParams(convo_obj, pool_size=(2,2), pool_strides=(1,1), pool_padding="same", pool_method="max")
        dense_obj = DenseParams(pool_obj, output_units=units)

        keras_obj = KerasParams([convo_obj,pool_obj,dense_obj])
        keras_obj.dense_input = (keras_obj.features_extractor(convo_obj.mock_image)[1].numpy() > 0.5).astype(int)
        keras_obj.dense_answer = (keras_obj.features_extractor(convo_obj.mock_image)[2].numpy() > 0.5).astype(int)
        dense_answer = (keras_obj.model.layers[2](dense_obj.dense_input).numpy() > 0.5).astype(int)
        expected_spike_count = dense_answer.sum()

        result, graph = run_whetstone_to_fugu_utility(convo_obj.mock_image, self.basep, self.bits, keras_obj.model)
        calculated_spike_count = len(get_dense_neurons_result_only(result,graph).index)
        assert expected_spike_count == calculated_spike_count

    def test_utility_keras_flattened_dense_layer(self):
        convo_obj = ConvolutionParams(image_height=5, image_width=5, nChannels=2, kernel_height=2, kernel_width=2, nFilters=3, biases=None)
        convo_obj.biases = convo_obj.get_random_biases_within_answer_range()
        convo_obj._set_convolution_answer_boolean()
        pool_obj = PoolingParams(convo_obj, pool_size=(2,2), pool_strides=(1,1), pool_padding="same", pool_method="max")
        dense_obj = DenseParams(pool_obj, output_units=4, output_shape=(1,np.prod(pool_obj.output_shape)))

        model = Sequential()
        model.add(Conv2D(convo_obj.nFilters,
                            (convo_obj.kernel_height, convo_obj.kernel_width),
                            strides=convo_obj.strides,
                            padding=convo_obj.mode,
                            activation=None,
                            use_bias=True,
                            input_shape=convo_obj.mock_image.shape[1:],
                            name=str(1),
                            kernel_initializer=initializers.constant(np.flip(convo_obj.filters,(0,1))),
                            bias_initializer=initializers.constant(np.array(convo_obj.biases).reshape((convo_obj.nFilters,)))))
        model.add(MaxPooling2D(pool_size=pool_obj.pool_size, strides=pool_obj.pool_strides, padding=pool_obj.pool_padding, name=str(2)))
        model.add(Flatten())
        model.add(Dense(units=dense_obj.output_units,
                            use_bias=True,
                            kernel_initializer=ArraySequence(dense_obj.weights),
                            bias_initializer=ArraySequence(dense_obj.biases),
                            name=str(4)))

        dense_answer = (model.layers[3](dense_obj.dense_input.flatten().reshape(1,-1)).numpy() > 0.5).astype(int)
        expected_spike_count = dense_answer.sum()

        result, graph = run_whetstone_to_fugu_utility(convo_obj.mock_image, self.basep, self.bits, model)
        calculated_spike_count = len(get_dense_neurons_result_only(result,graph).index)
        assert expected_spike_count == calculated_spike_count

    @pytest.mark.xfail(reason="Not implemented.")
    def test_simple_keras_dense_layer(self):
        assert False

    @pytest.mark.xfail(reason="Not implemented.")
    def test_keras_dense_layer(self):
        assert False

    @pytest.mark.xfail(reason="Not implemented.")
    def test_keras_dense_layer_with_biases_without_activation(self):
        assert False

    @pytest.mark.xfail(reason="Not implemented.")
    def test_keras_dense_layer_without_biases_without_activation(self):
        assert False

    @pytest.mark.xfail(reason="Not implemented.")
    def test_keras_dense_layer_with_biases_with_activation(self):
        assert False

    @pytest.mark.xfail(reason="Not implemented.")
    def test_keras_dense_layer_without_biases_with_activation(self):
        assert False

def get_neuron_numbers(name_prefix, graph):
    neuron_numbers = []
    for key in graph.nodes.keys():
        if key.startswith(name_prefix):
            neuron_numbers.append(graph.nodes[key]['neuron_number'])

    return np.array(neuron_numbers)

def get_dense_neurons_result_only(result, graph):
    dense_neuron_numbers = get_neuron_numbers('dense_layer_',graph)
    sub_result = result[result['neuron_number'].isin(dense_neuron_numbers)]
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
