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
from fugu.utils.whetstone import whetstone_2_fugu

import pandas as pd
from tensorflow import constant as tf_constant
from tensorflow.keras import Model, initializers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, BatchNormalization

from ..helpers import ConvolutionParams, PoolingParams, KerasParams

Spiking_BRelu = pytest.importorskip("whetstone.layers", reason=f"Whetstone package not installed. Skipping test file {__file__} because of module dependency.").Spiking_BRelu
layer_utils = pytest.importorskip("whetstone.utils", reason=f"Whetstone package not installed. Skipping test file {__file__} because of module dependency.").layer_utils

class Test_Whetstone_2_Fugu_MaxPoolingLayer:
    def setup_method(self):
        self.basep = 4
        self.bits = 3

        image_height, image_width, nChannels = 3, 3, 2
        kernel_height, kernel_width, nFilters = 2, 2, 3

        self.mode = "same"
        self.strides = (1,1)
        self.padding = "same"

        input_shape = (image_height,image_width,nChannels)
        init_kernel = generate_keras_kernel(kernel_height,kernel_width,nFilters,nChannels)
        init_bias = np.array([-320., -653., -786.]).reshape((nFilters,))
        kernel_initializer = initializers.constant(np.flip(init_kernel,(0,1))) # [METHOD 2] keras doesn't flip the filter during the convolution; so force the array flip manually.
        bias_initializer = initializers.constant(init_bias)

    @pytest.mark.xfail(reason="Not implemented.")
    def test_simple_keras_max_pooling_layer(self):
        input_shape = (3,3,1)
        strides = (1,1)
        init_kernel = np.reshape(np.flip(np.arange(1,5)),(1,2,2,1)) # [METHOD 2] keras doesn't flip the filter during the convolution; so force the array flip manually.
        init_bias = np.zeros((1,))
        kernel_initializer = initializers.constant(init_kernel)
        bias_initializer = initializers.constant(init_bias)
        nSpikes = 1

        model = Sequential()
        model.add(MaxPooling2D(pool_size=(2, 2),strides=(1,1),input_shape=input_shape))
        mock_image = np.arange(1,10,dtype=float).reshape(1,3,3,1)
        calculated = model.layers[0](mock_image)[0,:,:,0].numpy().tolist()
        assert False

    def test_keras_max_pooling_layer_same_padding_strides_11_pool_size_22(self):
        convo_obj = ConvolutionParams(biases=np.array([-471., -1207., -1943.]))
        pool_obj = PoolingParams(convo_obj, pool_strides=(2,2), pool_padding="same")

        keras_obj = KerasParams([convo_obj,pool_obj])
        expected_spike_count = (keras_obj.features_extractor(convo_obj.mock_image)[1].numpy() > 0.5).astype(int).sum()

        result = run_whetstone_to_fugu_utility(convo_obj.mock_image, self.basep, self. bits, keras_obj.model)
        calculated_spike_count = len(result[result['time'] > 1].index)

        assert expected_spike_count == calculated_spike_count

    @pytest.mark.xfail(reason="Not implemented.")
    def test_keras_max_pooling_layer_same_padding_strides_12_pool_size_22(self):
        assert False

    @pytest.mark.xfail(reason="Not implemented.")
    def test_keras_max_pooling_layer_same_padding_strides_21_pool_size_22(self):
        assert False

    @pytest.mark.xfail(reason="Not implemented.")
    def test_keras_max_pooling_layer_same_padding_strides_22_pool_size_22(self):
        assert False

    def test_keras_max_pooling_layer_same_padding_strides_12_pool_size_32(self):
        convo_obj = ConvolutionParams(image_height=5, image_width=5, nChannels=2, kernel_height=2, kernel_width=2, nFilters=3, biases=None)
        convo_obj.biases = convo_obj.get_random_biases_within_answer_range()
        convo_obj._set_convolution_answer_boolean()
        pool_obj = PoolingParams(convo_obj, pool_size=(2,3), pool_strides=(1,1), pool_padding="same", pool_method="max")

        keras_obj = KerasParams([convo_obj,pool_obj])
        keras_pool_result = (keras_obj.features_extractor(convo_obj.mock_image)[1].numpy() > 0.5).astype(int)
        expected_spike_count = keras_pool_result.sum()

        result = run_whetstone_to_fugu_utility(convo_obj.mock_image, self.basep, self. bits, keras_obj.model)
        calculated_spike_count = len(result[result['time'] > 1].index)

        assert expected_spike_count == calculated_spike_count

    @pytest.mark.parametrize("pool_size", [(1,2),(2,1),(2,2),(2,3),(3,2),(3,3)])
    @pytest.mark.parametrize("pool_strides", [(1,1),(1,2),(2,1),(2,2),(1,3),(3,1),(2,3),(3,2),(3,3)])
    @pytest.mark.parametrize("pool_padding", ["same", "valid"])
    @pytest.mark.parametrize("pool_method", ["max", "average"])
    @pytest.mark.parametrize("nChannels,nFilters", [(2,3), (1,1)])
    def test_keras_max_pooling_layer(self, pool_size, pool_strides, pool_padding, pool_method, nChannels, nFilters):

        convo_obj = ConvolutionParams(image_height=5, image_width=5, nChannels=nChannels, kernel_height=2, kernel_width=2, nFilters=nFilters, biases=None)
        convo_obj.biases = convo_obj.get_random_biases_within_answer_range()
        convo_obj._set_convolution_answer_boolean()
        pool_obj = PoolingParams(convo_obj, pool_size=pool_size, pool_strides=pool_strides, pool_padding=pool_padding, pool_method=pool_method)

        keras_obj = KerasParams([convo_obj,pool_obj])
        expected_spike_count = (keras_obj.features_extractor(convo_obj.mock_image)[1].numpy() > 0.5).astype(int).sum()

        result = run_whetstone_to_fugu_utility(convo_obj.mock_image, self.basep, self. bits, keras_obj.model)
        calculated_spike_count = len(result[result['time'] > 1].index)

        assert expected_spike_count == calculated_spike_count

def run_whetstone_to_fugu_utility(mock_image, basep, bits, keras_model):
    scaffold = Scaffold()
    scaffold.add_brick(BaseP_Input(mock_image,p=basep,bits=bits,collapse_binary=False,name="I",time_dimension=False),"input")
    scaffold = whetstone_2_fugu(keras_model,basep,bits,scaffold=scaffold)
    scaffold.lay_bricks()
    scaffold.summary(verbose=1)
    backend = snn_Backend()
    backend_args = {}
    backend.compile(scaffold, backend_args)
    result = backend.run(5)
    return result


class Test_Whetstone_2_Fugu_AveragePoolingLayer:
    @pytest.mark.xfail(reason="Not implemented.")
    def test_simple_keras_average_pooling_layer(self):
        assert False

    @pytest.mark.xfail(reason="Not implemented.")
    def test_keras_average_pooling_layer_same_padding_strides_11_pool_size_22(self):
        assert False

    @pytest.mark.xfail(reason="Not implemented.")
    def test_keras_average_pooling_layer_same_padding_strides_12_pool_size_22(self):
        assert False

    @pytest.mark.xfail(reason="Not implemented.")
    def test_keras_average_pooling_layer_same_padding_strides_21_pool_size_22(self):
        assert False

    @pytest.mark.xfail(reason="Not implemented.")
    def test_keras_average_pooling_layer_same_padding_strides_22_pool_size_22(self):
        assert False

    @pytest.mark.xfail(reason="Not implemented.")
    def test_keras_average_pooling_layer_same_padding_strides_12_pool_size_32(self):
        assert False
