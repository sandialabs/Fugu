import numpy as np
import pytest

from fugu.backends import snn_Backend
from fugu.bricks.keras_convolution_bricks import keras_convolution_2d as convolution_2d
from fugu.bricks.dense_bricks import dense_layer_1d, dense_layer_2d
from fugu.bricks.input_bricks import BaseP_Input, Vector_Input
from fugu.bricks.pooling_bricks import pooling_1d, pooling_2d
from fugu.scaffold import Scaffold
from fugu.utils.whetstone import whetstone_2_fugu

import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, BatchNormalization
from whetstone.layers import Spiking_BRelu
from whetstone.utils import layer_utils

class Test_Whetstone_2_Fugu_PoolingLayer:
    @pytest.mark.xfail(reason="Not implemented.")
    def test_simple_keras_max_pooling_layer(self):
        from tensorflow.keras import Model, initializers
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