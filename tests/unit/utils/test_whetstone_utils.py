import numpy as np
import pytest
from scipy.signal import convolve2d

from fugu.backends import snn_Backend
from fugu.bricks.convolution_bricks import convolution_1d, convolution_2d
from fugu.bricks.dense_bricks import dense_layer_1d, dense_layer_2d
from fugu.bricks.input_bricks import BaseP_Input, Vector_Input
from fugu.bricks.pooling_bricks import pooling_1d, pooling_2d
from fugu.scaffold import Scaffold
from fugu.utils.whetstone import whetstone_2_fugu

import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from whetstone.utils import layer_utils

def mock_keras_model(filename):
    '''
        Monkeypatch (i.e., mock) for a keras model.
    '''
    # 28x28 image with 3 (RGB) channels
    input_shape = (28,28,3)
    model = Sequential()
    model.add(Conv2D(3, (7, 7), padding='same', activation=None, use_bias=True, input_shape=input_shape))
    return model

def keras_mnist_model_norm_off():
    import os
    return f"{os.path.dirname(__file__)}/data/model_adaptive_mnist_normalization_off.keras"

def keras_custom_model_inference():
    from tensorflow.keras import Model, initializers
    input_shape = (2,2,1)
    init_kernel = np.reshape(np.repeat(np.arange(1,5),3),(1,2,2,3))
    init_bias = np.zeros((3,))
    kernel_initializer = initializers.constant(init_kernel)
    bias_initializer = initializers.constant(init_bias)

    model = Sequential()
    model.add(Conv2D(3, (2, 2), padding='same', activation=None, use_bias=True, input_shape=input_shape, name="one", kernel_initializer=kernel_initializer, bias_initializer=bias_initializer))

    # feature_extractor = Model(inputs=model.inputs, outputs=model.get_layer(name="one").output)
    feature_extractor = Model(inputs=model.inputs, outputs=[layer.output for layer in model.layers])
    x = np.ones((1,2,2,1))
    features = feature_extractor(x)

class Test_Whetstone_2_Fugu_Normalization_Off:
    @pytest.mark.skip(reason="Not implemented.")
    def test_layers(self):
        model = layer_utils.load_model(keras_mnist_model_norm_off())

        basep = 3
        bits = 3
        scaffold = Scaffold()
        scaffold.add_brick(Vector_Input(np.ones((28,28)), name="Input0"),"input")
        scaffold = whetstone_2_fugu(model,basep,bits,scaffold=scaffold)
        self.graph = scaffold.lay_bricks()

        # test against layer names
        #  - for each layer, 
        #       - convolution layer
        #           - input shape, output shape, weights (filters), bias (thresholds), brick names, and number of bricks per convolution layer (to make sure channels are handled properly)
        #       - pooling layer
        #           - pool size, strides, brick names, and number of bricks per pooling laer (to make sure channels are handled properly)
        #       - dense layer
        #           - output shape, weights, bias (thresholds), and brick names

        assert False

    @pytest.mark.xfail(reason="Not implemented.")
    def test_mock_keras_model(self, monkeypatch):
        # monkeypatch.setattr("whetstone.utils.layer_utils.load_model", mock_keras_model)
        monkeypatch.setattr("whetstone.utils.layer_utils.load_model", mock_keras_model)
        model = layer_utils.load_model(keras_mnist_model_norm_off())
        keras_custom_model_inference()
        basep = 3
        bits = 3
        scaffold = Scaffold()
        scaffold.add_brick(Vector_Input(np.ones((28,28)), name="Input0"),"input")
        scaffold = whetstone_2_fugu(model,basep,bits,scaffold=scaffold)
        self.graph = scaffold.lay_bricks()        

        assert False

class Test_Whetstone_2_Fugu_Normalization_On:
    @pytest.mark.xfail(reason="Not implemented.")
    def test_layers(self):
        assert False

class Test_Keras_Conv2d:
    def test_simple_conv2d(self):
        from tensorflow.keras import Model, initializers
        
        input_shape = (2,2,1)
        init_kernel = np.reshape(np.arange(1,5),(1,2,2,1))
        init_bias = np.zeros((1,))
        kernel_initializer = initializers.constant(init_kernel)
        bias_initializer = initializers.constant(init_bias)

        model = Sequential()
        model.add(Conv2D(1, (2, 2), padding='same', activation=None, use_bias=True, input_shape=input_shape, name="one", kernel_initializer=kernel_initializer, bias_initializer=bias_initializer))

        # feature_extractor = Model(inputs=model.inputs, outputs=model.get_layer(name="one").output)
        feature_extractor = Model(inputs=model.inputs, outputs=[layer.output for layer in model.layers])
        image = np.ones((1,2,2,1))
        calculated = np.flip(feature_extractor(image)[0,:,:,0].numpy()).tolist()
        expected = [[1.,3.],[4.,10.]]
        assert expected == calculated

    @pytest.mark.xfail(reason="Not implemented.")
    def test_multichannel_conv2d(self):
        from tensorflow.keras import Model, initializers
        
        nChannels = 2
        input_shape = (2,2,nChannels)
        init_kernel = np.reshape(np.repeat(np.arange(1,5),nChannels),(1,2,2,nChannels))
        init_bias = np.zeros((1,))
        kernel_initializer = initializers.constant(init_kernel)
        bias_initializer = initializers.constant(init_bias)

        model = Sequential()
        model.add(Conv2D(1, (2, 2), padding='same', activation=None, use_bias=True, input_shape=input_shape, name="one", kernel_initializer=kernel_initializer, bias_initializer=bias_initializer))

        # feature_extractor = Model(inputs=model.inputs, outputs=model.get_layer(name="one").output)
        feature_extractor = Model(inputs=model.inputs, outputs=[layer.output for layer in model.layers])
        
        image = np.ones((1,2,2,nChannels))
        calculated = np.flip(feature_extractor(image)[0,:,:,:].numpy()).tolist()
        expected = [[2.,6.],[8.,20.]]
        assert expected == calculated

    def test_multifilter_conv2d(self):
        from tensorflow.keras import Model, initializers
        
        nFilters = 3
        input_shape = (2,2,1)
        init_kernel = np.reshape(np.repeat(np.arange(1,5),nFilters),(1,2,2,nFilters))
        init_bias = np.zeros((nFilters,))
        kernel_initializer = initializers.constant(init_kernel)
        bias_initializer = initializers.constant(init_bias)

        model = Sequential()
        model.add(Conv2D(nFilters, (2, 2), padding='same', activation=None, use_bias=True, input_shape=input_shape, name="one", kernel_initializer=kernel_initializer, bias_initializer=bias_initializer))

        # feature_extractor = Model(inputs=model.inputs, outputs=model.get_layer(name="one").output)
        feature_extractor = Model(inputs=model.inputs, outputs=[layer.output for layer in model.layers])
        
        image = np.ones((1,2,2,1))
        calculated = np.flip(feature_extractor(image)[0,:,:,:].numpy()).tolist()
        expected = [[[1.,1.,1.],[3.,3.,3.]],[[4.,4.,4.],[10.,10.,10.]]]
        assert expected == calculated

    def test_multifilter_multichannel_conv2d(self):
        from tensorflow.keras import Model, initializers
        
        nFilters = 3
        nChannels = 2
        input_shape = (2,2,nChannels)
        init_kernel = np.reshape(np.repeat(np.arange(1,5),nChannels*nFilters),(1,2,2,nFilters*nChannels))
        init_bias = np.zeros((nFilters,))
        kernel_initializer = initializers.constant(init_kernel)
        bias_initializer = initializers.constant(init_bias)

        model = Sequential()
        model.add(Conv2D(nFilters, (2, 2), padding='same', activation=None, use_bias=True, input_shape=input_shape, name="one", kernel_initializer=kernel_initializer, bias_initializer=bias_initializer))

        # feature_extractor = Model(inputs=model.inputs, outputs=model.get_layer(name="one").output)
        feature_extractor = Model(inputs=model.inputs, outputs=[layer.output for layer in model.layers])
        
        image = np.ones((1,2,2,nChannels))
        calculated = np.flip(feature_extractor(image)[0,:,:,:].numpy()).tolist()

        # expected result
        ans = nChannels * convolve2d(image[0,:,:,0],init_kernel[0,:,:,0],mode='same')
        expected = np.repeat(ans,nFilters).reshape(2,2,nFilters).tolist()
        assert expected == calculated