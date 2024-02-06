# isort: skip_file
# fmt: off
import numpy as np
import pytest
from scipy.signal import convolve2d

from fugu.backends import snn_Backend
from fugu.bricks.keras_convolution_bricks import keras_convolution_2d_4dinput as convolution_2d
from fugu.bricks.dense_bricks import dense_layer_1d, dense_layer_2d
from fugu.bricks.input_bricks import BaseP_Input, Vector_Input
from fugu.bricks.pooling_bricks import pooling_1d, pooling_2d
from fugu.scaffold import Scaffold
from fugu.utils.keras_helpers import keras_convolve2d, keras_convolve2d_4dinput, generate_keras_kernel, generate_mock_image, keras_convolution2d_output_shape, keras_convolution2d_output_shape_4dinput
from fugu.utils.whetstone_conversion import whetstone_2_fugu

import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, BatchNormalization

Spiking_BRelu = pytest.importorskip("whetstone.layers", reason=f"Whetstone package not installed. Skipping test file {__file__} because of module dependency.").Spiking_BRelu
layer_utils = pytest.importorskip("whetstone.utils", reason=f"Whetstone package not installed. Skipping test file {__file__} because of module dependency.").layer_utils

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

def keras_fixed_model_inference(biases):
    '''
        Keras 2D convolution answer with biases = [0,0,0], mode="same", and strides=(1,1)
        [[328., 364., 210.],  [[ 808.,  908.,  498.],  [[1288., 1452.,  786.],
         [436., 472., 270.],   [1108., 1208.,  654.],   [1780., 1944., 1038.],
         [299., 321., 180.]],  [ 683.,  737.,  396.]],  [1067., 1153.,  612.]], 
    '''
    from tensorflow.keras import Model, initializers
    image_height, image_width = 3, 3
    kernel_height, kernel_width = 2, 2
    nChannels = 2
    nFilters = 3
    mode = "same"
    strides = (1,1)

    input_shape = (image_height,image_width,nChannels)
    init_kernel = generate_keras_kernel(kernel_height,kernel_width,nFilters,nChannels)
    init_bias = np.array(biases).reshape((nFilters,))
    kernel_initializer = initializers.constant(np.flip(init_kernel,(0,1))) # [METHOD 2] keras doesn't flip the filter during the convolution; so force the array flip manually.
    bias_initializer = initializers.constant(init_bias)
            
    model = Sequential()
    model.add(Conv2D(nFilters, (kernel_height, kernel_width), strides=strides, padding=mode, activation=None, use_bias=True, input_shape=input_shape, name="one", kernel_initializer=kernel_initializer, bias_initializer=bias_initializer))

    # feature_extractor = Model(inputs=model.inputs, outputs=model.get_layer(name="one").output)
    feature_extractor = Model(inputs=model.inputs, outputs=[layer.output for layer in model.layers])
    mock_image = generate_mock_image(image_height, image_width, nChannels)
    features = feature_extractor(mock_image)
    return features, model

def keras_custom_model_inference(biases, image_shape, kernel_shape, nChannels, nFilters, mode, strides):
    '''
        Keras 2D convolution answer with biases = [0,0,0], mode="same", and strides=(1,1)
        [[328., 364., 210.],  [[ 808.,  908.,  498.],  [[1288., 1452.,  786.],
         [436., 472., 270.],   [1108., 1208.,  654.],   [1780., 1944., 1038.],
         [299., 321., 180.]],  [ 683.,  737.,  396.]],  [1067., 1153.,  612.]], 
    '''
    from tensorflow.keras import Model, initializers
    image_height, image_width = image_shape
    kernel_height, kernel_width = kernel_shape

    input_shape = (image_height,image_width,nChannels)
    init_kernel = generate_keras_kernel(kernel_height,kernel_width,nFilters,nChannels)
    init_bias = np.array(biases).reshape((nFilters,))
    kernel_initializer = initializers.constant(np.flip(init_kernel,(0,1))) # [METHOD 2] keras doesn't flip the filter during the convolution; so force the array flip manually.
    bias_initializer = initializers.constant(init_bias)
            
    model = Sequential()
    model.add(Conv2D(nFilters, (kernel_height, kernel_width), strides=strides, padding=mode, activation=None, use_bias=True, input_shape=input_shape, name="one", kernel_initializer=kernel_initializer, bias_initializer=bias_initializer))

    # feature_extractor = Model(inputs=model.inputs, outputs=model.get_layer(name="one").output)
    feature_extractor = Model(inputs=model.inputs, outputs=[layer.output for layer in model.layers])
    mock_image = generate_mock_image(image_height, image_width, nChannels)
    features = feature_extractor(mock_image)
    return features, model

@pytest.fixture(params=["same","valid"])
def convolution_mode(request):
    return request.param

@pytest.fixture
def convolution_strides():
    return (1,1)

@pytest.fixture
def convolution_params(convolution_mode, convolution_strides):
    return convolution_mode, convolution_strides 

@pytest.fixture
def image_channels():
    return 1

@pytest.fixture
def image_shape():
    return 3, 3

@pytest.fixture 
def image_params(image_shape, image_channels):
    return image_shape[0], image_shape[1], image_channels

@pytest.fixture
def kernel_shape():
    return 2, 2

@pytest.fixture
def kernel_filters():
    return 1

@pytest.fixture
def kernel_params(kernel_shape, kernel_filters):
    return kernel_shape[0], kernel_shape[1], kernel_filters

@pytest.fixture
def custom_setup(image_params,kernel_params,convolution_params):
    image_height, image_width, nChannels = image_params
    kernel_height, kernel_width, nFilters = kernel_params
    mode, strides = convolution_params

@pytest.mark.whetstone
class Test_Whetstone_2_Fugu_ConvolutionLayer:
    @pytest.mark.xfail(run=False,reason="Not implemented. And takes to long for 'xfail'.")
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

    @pytest.mark.parametrize("convolution_mode", ["same"])
    @pytest.mark.parametrize("bias", [(-22),(-23),(-24),(-33),(-36),(-42),(-52),(-53),(-59),(-63)])
    def test_temporary(self,image_params,kernel_params,convolution_params,bias):
        from tensorflow.keras import Model, initializers
        image_height, image_width, nChannels = image_params
        kernel_height, kernel_width, nFilters = kernel_params
        mode, strides = convolution_params

        features, model = keras_custom_model_inference(bias,(image_height,image_width),(kernel_height,kernel_width),nChannels,nFilters,mode,strides)
        self.basep = 3
        self.bits = 3
        self.pvector = generate_mock_image(image_height,image_width,nChannels).astype(float)
        result = self.run_whetstone_to_fugu_utility(model)

        expected_spikes = (features[0].numpy() > 0.5).sum()
        calculated_spikes = len(result.index)
        assert expected_spikes == calculated_spikes

    @pytest.mark.parametrize("biases, expected_spikes", [([-320., -653., -786.], 19),([-471.,-1207.,-1943.], 3), ([-472.,-1208.,-1944.], 0)])
    def test_mock_keras_model(self, monkeypatch,biases,expected_spikes):
        # monkeypatch.setattr("whetstone.utils.layer_utils.load_model", mock_keras_model)
        # monkeypatch.setattr("whetstone.utils.layer_utils.load_model", mock_keras_model)
        # model = layer_utils.load_model(keras_mnist_model_norm_off())
        features, model = keras_fixed_model_inference(biases)

        calculated_spikes = (features.numpy() > 0.5).sum()

        assert expected_spikes == calculated_spikes

    @pytest.mark.parametrize("bias,nSpikes", [(-22,9),(-23,8),(-24,7),(-33,6),(-36,5),(-42,4),(-52,3),(-53,2),(-59,1),(-63,0)])
    def test_whetstone_2_fugu_conv2d_layer(self,bias,nSpikes):
        '''
            [[1,2,3]                 [[ 4,11,18, 9]
             [4,5,6]  *  [[1,2]  =    [18,37,47,21]
             [7,8,9]]     [3,4]]      [36,67,77,33]
                                      [14,23,26, 9]]

            *** Note ***
            [METHOD 1]
            If you don't flip the kernel for Keras then don't "flip" the kernel when doing convolution by hand. Moreover this result is equivalent 
            to performing scipy.signal.convolve2d([[1,2,3],[4,5,6],[7,8,9]],np.flip( [[1,2],[3,4]] ), mode="full").

            [METHOD 2]
            Otherwise, "flip" the kernel (filter) for keras and then perform the traditional convolution practice when calculating convolution by hand.
            The below result is equivalent to scipy.signal.convolve2d([[1,2,3],[4,5,6],[7,8,9]],[[1,2],[3,4]], mode="full")

            [[1,2,3]                 [[ 1, 4, 7, 6]
             [4,5,6]  *  [[1,2]  =    [ 7,23,33,24]
             [7,8,9]]     [3,4]]      [19,53,63,42]
                                      [21,52,59,36]]
        '''
        from tensorflow.keras import Model, initializers
        image_height, image_width = 3, 3
        kernel_height, kernel_width = 2, 2
        nChannels = 1
        nFilters = 1
        strides = (1,1)
        mode = "same"

        input_shape = (image_height,image_width,nChannels)
        mock_image = generate_mock_image(image_height,image_width,nChannels).astype(float)
        init_kernel = generate_keras_kernel(kernel_height,kernel_width,nFilters,nChannels)
        init_bias = bias*np.ones((nFilters,))
        kernel_initializer = initializers.constant(np.flip(init_kernel,(0,1))) # [METHOD 2] keras doesn't flip the filter during the convolution; so force the array flip manually.
        bias_initializer = initializers.constant(init_bias)

        model = Sequential()
        model.add(Conv2D(nFilters, (kernel_height, kernel_width), strides=strides, padding=mode, activation=None, use_bias=True, input_shape=input_shape, name="one", kernel_initializer=kernel_initializer, bias_initializer=bias_initializer))
        model.add(Spiking_BRelu(input_shape=(3,3),sharpness=1.0,name="spike"))
        calculated = model.layers[0](mock_image)[0,:,:,0].numpy().tolist() # gives the output of the input through this layer only
        feature_extractor = Model(inputs=model.inputs, outputs=[layer.output for layer in model.layers]) # gives cumalative output of the input through this and previous layers.

        self.basep = 3
        self.bits = 3
        self.pvector = mock_image
        self.pshape = self.pvector.shape
        self.filters = init_kernel
        self.filters_shape = self.filters.shape
        self.strides = strides
        self.nFilters = nFilters
        self.mode = mode
        result = self.run_whetstone_to_fugu_utility(model)

        expected_spike_count = (feature_extractor(mock_image)[0][0,:,:,0].numpy() > 0.5).sum()
        calculated_spike_count = len(result.index)
        assert expected_spike_count == calculated_spike_count

    def test_explicit_whetstone_2_fugu_conv2d_layer(self):
        '''
            [[1,2,3]                  [[ 4,11,18, 9]
             [4,5,6]  *  [[1,2]  =     [18,37,47,21]
             [7,8,9]]     [3,4]]       [36,67,77,33]
                                       [14,23,26, 9]]

            *** Note ***
            [METHOD 1]
            If you don't flip the kernel for Keras then don't "flip" the kernel when doing convolution by hand. Moreover this result is equivalent 
            to performing scipy.signal.convolve2d([[1,2,3],[4,5,6],[7,8,9]],np.flip( [[1,2],[3,4]] ), mode="full").

            [METHOD 2]
            Otherwise, "flip" the kernel (filter) for keras and then perform the traditional convolution practice when calculating convolution by hand.
            The below result is equivalent to scipy.signal.convolve2d([[1,2,3],[4,5,6],[7,8,9]],[[1,2],[3,4]], mode="full")

            [[1,2,3]                  [[ 1, 4, 7, 6]
             [4,5,6]  *  [[1,2]  =     [ 7,23,33,24]
             [7,8,9]]     [3,4]]       [19,53,63,42]
                                       [21,52,59,36]]
        '''
        from tensorflow.keras import Model, initializers
        image_height, image_width = 3, 3
        kernel_height, kernel_width = 2, 2
        nChannels = 1
        nFilters = 1
        strides = (1,1)
        mode = "same"
        nSpikes = 2

        input_shape = (image_height,image_width,nChannels)
        mock_image = generate_mock_image(image_height,image_width,nChannels).astype(float)
        init_kernel = generate_keras_kernel(kernel_height,kernel_width,nFilters,nChannels)
        init_bias = -52.6*np.ones((nFilters,))
        kernel_initializer = initializers.constant(np.flip(init_kernel,(0,1))) # [METHOD 2] keras doesn't flip the filter during the convolution; so force the array flip manually.
        bias_initializer = initializers.constant(init_bias)

        model = Sequential()
        model.add(Conv2D(nFilters, (kernel_height, kernel_width), strides=strides, padding=mode, activation=None, use_bias=True, input_shape=input_shape, name="one", kernel_initializer=kernel_initializer, bias_initializer=bias_initializer))
        model.add(Spiking_BRelu(input_shape=(3,3),sharpness=1.0,name="spike"))
        mock_image = generate_mock_image(image_height,image_width,nChannels).astype(float)
        calculated = model.layers[0](mock_image)[0,:,:,0].numpy().tolist() # gives the output of the input through this layer only
        feature_extractor = Model(inputs=model.inputs, outputs=[layer.output for layer in model.layers]) # gives cumalative output of the input through this and previous layers.

        self.basep = 3
        self.bits = 3
        self.pvector = mock_image
        self.pshape = self.pvector.shape
        self.filters = init_kernel
        self.filters_shape = self.filters.shape
        self.strides = strides
        self.nFilters = nFilters
        self.mode = mode
        result = self.run_whetstone_to_fugu_utility(model)

        expected_spike_count = (feature_extractor(mock_image)[0][0,:,:,0].numpy() > 0.5).sum()
        calculated_spike_count = len(result.index)
        assert expected_spike_count == calculated_spike_count

    def test_explicit_whetstone_2_fugu_conv2d_layer_multichannel_multifilter_mode_same_strides_11(self):
        '''
            Mode="same", Strides=(1,1), Convolution answer is with all zero biases.

            Image:
                   Channel 1         Channel 2
                [[1., 2., 3.],  [[10., 11., 12.],
                 [4., 5., 6.],   [13., 14., 15.],
                 [7., 8., 9.]]   [16., 17., 18.]]

            Filters:
              Filter 1
                Channel 1   Channel 2
                [[ 1,  2],  [[5, 6],
                 [ 3,  4]]   [7, 8]]

              Filter 2
                Channel 1   Channel 2
                [[ 9, 10],  [[13, 14],
                 [11, 12]]   [15, 16]]

              Filter 3
                Channel 1   Channel 2
                [[17, 18],  [[21, 22],
                 [19, 20]]   [23, 24]]

            Convolution Answer:
                  Filter 1 Out              Filter 2 Out              Filter 3 Out
            [[ 328.,  364.,  210.],   [[ 808.,  908.,  498.],   [[1288., 1452.,  786.],
             [ 436.,  472.,  270.],    [1108., 1208.,  654.],    [1780., 1944., 1038.],
             [ 299.,  321.,  180.]]    [ 683.,  737.,  396.]]    [1067., 1153.,  612.]]
        '''
        from tensorflow.keras import Model, initializers
        image_height, image_width = 3, 3
        kernel_height, kernel_width = 2, 2
        nChannels = 2
        nFilters = 3
        input_shape = (image_height,image_width,nChannels)
        strides = (1,1)
        init_kernel = generate_keras_kernel(kernel_height,kernel_width,nFilters,nChannels)
        init_bias = np.array([-320., -653., -786.]).reshape((nFilters,))
        kernel_initializer = initializers.constant(np.flip(init_kernel,(0,1))) # [METHOD 2] keras doesn't flip the filter during the convolution; so force the array flip manually.
        bias_initializer = initializers.constant(init_bias)
        mock_image = generate_mock_image(image_height,image_width,nChannels).astype(float)
        mode = "same"
        nSpikes = 5 + 7 + 7
        basep = 3
        bits = 4

        model = Sequential()
        model.add(Conv2D(nFilters, (kernel_height, kernel_width), strides=strides, padding=mode, activation=None, use_bias=True, input_shape=input_shape, name="one", kernel_initializer=kernel_initializer, bias_initializer=bias_initializer))
        model.add(Spiking_BRelu(sharpness=1.0,name="spike"))
        feature_extractor = Model(inputs=model.inputs, outputs=[layer.output for layer in model.layers]) # gives cumalative output of the input through this and previous layers.

        self.basep = basep
        self.bits = bits
        self.pvector = mock_image
        self.pshape = self.pvector.shape
        self.filters = init_kernel
        self.filters_shape = self.filters.shape
        self.strides = strides
        self.nFilters = nFilters
        self.mode = mode
        result = self.run_whetstone_to_fugu_utility(model)

        expected_spike_count = (feature_extractor(mock_image)[0].numpy() > 0.5).sum()
        calculated_spike_count = len(result.index)
        assert expected_spike_count == calculated_spike_count

    def test_explicit_whetstone_2_fugu_conv2d_layer_multichannel_multifilter_mode_same_strides_12(self):
        '''
            mode="same", strides=(1,2). Answers are with biases equal to zero.

            Image:
                   Channel 1         Channel 2
                [[1., 2., 3.],  [[10., 11., 12.],
                 [4., 5., 6.],   [13., 14., 15.],
                 [7., 8., 9.]]   [16., 17., 18.]]

            Filters:
              Filter 1
                Channel 1   Channel 2
                [[ 1,  2],  [[5, 6],
                 [ 3,  4]]   [7, 8]]

              Filter 2
                Channel 1   Channel 2
                [[ 9, 10],  [[13, 14],
                 [11, 12]]   [15, 16]]

              Filter 3
                Channel 1   Channel 2
                [[17, 18],  [[21, 22],
                 [19, 20]]   [23, 24]]

            Convolution Answer:
            Filter 1 Out     Filter 2 Out     Filter 3 Out
            [[ 328.  210.]   [[ 808.  498.]   [[1288.  786.] 
             [ 436.  270.]    [1108.  654.]    [1780. 1038.] 
             [ 299.  180.]],  [ 683.  396.]],  [1067.  612.]]
        '''
        from tensorflow.keras import Model, initializers
        image_height, image_width = 3, 3
        kernel_height, kernel_width = 2, 2
        nChannels = 2
        nFilters = 3
        input_shape = (image_height,image_width,nChannels)
        strides = (1,2)
        init_kernel = generate_keras_kernel(kernel_height,kernel_width,nFilters,nChannels)
        init_bias = np.array([-320., -653., -786.]).reshape((nFilters,))
        kernel_initializer = initializers.constant(np.flip(init_kernel,(0,1))) # [METHOD 2] keras doesn't flip the filter during the convolution; so force the array flip manually.
        bias_initializer = initializers.constant(init_bias)
        mock_image = generate_mock_image(image_height,image_width,nChannels).astype(float)
        mode = "same"
        basep = 3
        bits = 4

        model = Sequential()
        model.add(Conv2D(nFilters, (kernel_height, kernel_width), strides=strides, padding=mode, activation=None, use_bias=True, input_shape=input_shape, name="one", kernel_initializer=kernel_initializer, bias_initializer=bias_initializer))
        model.add(Spiking_BRelu(sharpness=1.0,name="spike"))
        feature_extractor = Model(inputs=model.inputs, outputs=[layer.output for layer in model.layers]) # gives cumalative output of the input through this and previous layers.

        self.basep = basep
        self.bits = bits
        self.pvector = mock_image
        self.pshape = self.pvector.shape
        self.filters = init_kernel
        self.filters_shape = self.filters.shape
        self.strides = strides
        self.nFilters = nFilters
        self.mode = mode
        result = self.run_whetstone_to_fugu_utility(model)

        expected_num_spikes = (feature_extractor(mock_image)[0].numpy() > 0.5).sum()
        calculated_num_spikes = len(result.index)
        assert expected_num_spikes == calculated_num_spikes

    def test_explicit_whetstone_2_fugu_conv2d_layer_multichannel_multifilter_mode_same_strides_21(self):
        '''
            mode="same", strides=(2,1). Answers are with biases equal to zero.

            Image:
                   Channel 1         Channel 2
                [[1., 2., 3.],  [[10., 11., 12.],
                 [4., 5., 6.],   [13., 14., 15.],
                 [7., 8., 9.]]   [16., 17., 18.]]

            Filters:
              Filter 1
                Channel 1   Channel 2
                [[ 1,  2],  [[5, 6],
                 [ 3,  4]]   [7, 8]]

              Filter 2
                Channel 1   Channel 2
                [[ 9, 10],  [[13, 14],
                 [11, 12]]   [15, 16]]

              Filter 3
                Channel 1   Channel 2
                [[17, 18],  [[21, 22],
                 [19, 20]]   [23, 24]]

            Convolution Answer:
            Filter 1 Out           Filter 2 Out           Filter 3 Out
            [[ 328.  364.  210.]   [[ 808.  908.  498.]   [[1288. 1452.  786.] 
             [ 299.  321.  180.]],  [ 683.  737.  396.]],  [1067. 1153.  612.]]
        '''
        from tensorflow.keras import Model, initializers
        image_height, image_width = 3, 3
        kernel_height, kernel_width = 2, 2
        nChannels = 2
        nFilters = 3
        input_shape = (image_height,image_width,nChannels)
        strides = (2,1)
        init_kernel = generate_keras_kernel(kernel_height,kernel_width,nFilters,nChannels)
        init_bias = np.array([-320., -653., -786.]).reshape((nFilters,))
        kernel_initializer = initializers.constant(np.flip(init_kernel,(0,1))) # [METHOD 2] keras doesn't flip the filter during the convolution; so force the array flip manually.
        bias_initializer = initializers.constant(init_bias)
        mock_image = generate_mock_image(image_height,image_width,nChannels).astype(float)
        mode = "same"
        basep = 3
        bits = 4

        model = Sequential()
        model.add(Conv2D(nFilters, (kernel_height, kernel_width), strides=strides, padding=mode, activation=None, use_bias=True, input_shape=input_shape, name="one", kernel_initializer=kernel_initializer, bias_initializer=bias_initializer))
        model.add(Spiking_BRelu(sharpness=1.0,name="spike"))
        feature_extractor = Model(inputs=model.inputs, outputs=[layer.output for layer in model.layers]) # gives cumalative output of the input through this and previous layers.

        self.basep = basep
        self.bits = bits
        self.pvector = mock_image
        self.pshape = self.pvector.shape
        self.filters = init_kernel
        self.filters_shape = self.filters.shape
        self.strides = strides
        self.nFilters = nFilters
        self.mode = mode
        result = self.run_whetstone_to_fugu_utility(model)

        expected_num_spikes = (feature_extractor(mock_image)[0].numpy() > 0.5).sum()
        calculated_num_spikes = len(result.index)
        assert expected_num_spikes == calculated_num_spikes

    def test_explicit_whetstone_2_fugu_conv2d_layer_multichannel_multifilter_mode_same_strides_22(self):
        '''
            mode="same", strides=(2,2). Answers are with biases equal to zero.

            Image:
                   Channel 1         Channel 2
                [[1., 2., 3.],  [[10., 11., 12.],
                 [4., 5., 6.],   [13., 14., 15.],
                 [7., 8., 9.]]   [16., 17., 18.]]

            Filters:
              Filter 1
                Channel 1   Channel 2
                [[ 1,  2],  [[5, 6],
                 [ 3,  4]]   [7, 8]]

              Filter 2
                Channel 1   Channel 2
                [[ 9, 10],  [[13, 14],
                 [11, 12]]   [15, 16]]

              Filter 3
                Channel 1   Channel 2
                [[17, 18],  [[21, 22],
                 [19, 20]]   [23, 24]]

            Convolution Answer:
            Filter 1 Out    Filter 2 Out    Filter 3 Out
            [[328. 210.]    [[808. 498.]    [[1288.  786.] 
             [299. 180.]],   [683. 396.]],   [1067.  612.]]

        '''
        from tensorflow.keras import Model, initializers
        image_height, image_width = 3, 3
        kernel_height, kernel_width = 2, 2
        nChannels = 2
        nFilters = 3
        input_shape = (image_height,image_width,nChannels)
        strides = (2,2)
        init_kernel = generate_keras_kernel(kernel_height,kernel_width,nFilters,nChannels)
        init_bias = np.array([-320., -653., -786.]).reshape((nFilters,)) # Results in 1 spike from filter 1, 2 spikes from filter 2, 2 spikes from filter 3. Spikes occur when filter outputs are > 0.5.
        kernel_initializer = initializers.constant(np.flip(init_kernel,(0,1))) # [METHOD 2] keras doesn't flip the filter during the convolution; so force the array flip manually.
        bias_initializer = initializers.constant(init_bias)
        mock_image = generate_mock_image(image_height,image_width,nChannels).astype(float)
        mode = "same"
        basep = 3
        bits = 4

        model = Sequential()
        model.add(Conv2D(nFilters, (kernel_height, kernel_width), strides=strides, padding=mode, activation=None, use_bias=True, input_shape=input_shape, name="one", kernel_initializer=kernel_initializer, bias_initializer=bias_initializer))
        model.add(Spiking_BRelu(sharpness=1.0,name="spike"))
        feature_extractor = Model(inputs=model.inputs, outputs=[layer.output for layer in model.layers]) # gives cumalative output of the input through this and previous layers.

        self.basep = basep
        self.bits = bits
        self.pvector = mock_image
        self.pshape = self.pvector.shape
        self.filters = init_kernel
        self.filters_shape = self.filters.shape
        self.strides = strides
        self.nFilters = nFilters
        self.mode = mode
        result = self.run_whetstone_to_fugu_utility(model)

        expected_num_spikes = (feature_extractor(mock_image)[0].numpy() > 0.5).sum()
        calculated_num_spikes = len(result.index)
        assert expected_num_spikes == calculated_num_spikes

    def test_explicit_whetstone_2_fugu_conv2d_layer_multichannel_multifilter_mode_valid_strides_11(self):
        '''
            mode="valid", strides=(1,1). Answers are with biases equal to zero.

            Image:
                   Channel 1                Channel 2
                [[ 1.  2.  3.  4.  5.] , [[26. 27. 28. 29. 30.] 
                 [ 6.  7.  8.  9. 10.] ,  [31. 32. 33. 34. 35.] 
                 [11. 12. 13. 14. 15.] ,  [36. 37. 38. 39. 40.] 
                 [16. 17. 18. 19. 20.] ,  [41. 42. 43. 44. 45.] 
                 [21. 22. 23. 24. 25.]],  [46. 47. 48. 49. 50.]]

            Filters:
              Filter 1
                Channel 1   Channel 2
                [[ 1,  2],  [[5, 6],
                 [ 3,  4]]   [7, 8]]

              Filter 2
                Channel 1   Channel 2
                [[ 9, 10],  [[13, 14],
                 [11, 12]]   [15, 16]]

              Filter 3
                Channel 1   Channel 2
                [[17, 18],  [[21, 22],
                 [19, 20]]   [23, 24]]

            Convolution Answer:
               Filter 1 Out                Filter 2 Out                 Filter 3 Out
            [[ 772.  808.  844.  880.]   [[1828. 1928. 2028. 2128.]   [[2884. 3048. 3212. 3376.] 
             [ 952.  988. 1024. 1060.]    [2328. 2428. 2528. 2628.]    [3704. 3868. 4032. 4196.] 
             [1132. 1168. 1204. 1240.]    [2828. 2928. 3028. 3128.]    [4524. 4688. 4852. 5016.] 
             [1312. 1348. 1384. 1420.]],  [3328. 3428. 3528. 3628.]],  [5344. 5508. 5672. 5836.]]
        '''
        from tensorflow.keras import Model, initializers
        image_height, image_width = 5, 5
        kernel_height, kernel_width = 2, 2
        nChannels = 2
        nFilters = 3
        input_shape = (image_height,image_width,nChannels)
        strides = (1,1)
        init_kernel = generate_keras_kernel(kernel_height,kernel_width,nFilters,nChannels)
        init_bias = np.array([-320., -653., -786.]).reshape((nFilters,)) # Results in 1 spike from filter 1, 2 spikes from filter 2, 2 spikes from filter 3. Spikes occur when filter outputs are > 0.5.
        kernel_initializer = initializers.constant(np.flip(init_kernel,(0,1))) # [METHOD 2] keras doesn't flip the filter during the convolution; so force the array flip manually.
        bias_initializer = initializers.constant(init_bias)
        mock_image = generate_mock_image(image_height,image_width,nChannels).astype(float)
        mode = "valid"
        basep = 3
        bits = 4

        model = Sequential()
        model.add(Conv2D(nFilters, (kernel_height, kernel_width), strides=strides, padding=mode, activation=None, use_bias=True, input_shape=input_shape, name="one", kernel_initializer=kernel_initializer, bias_initializer=bias_initializer))
        feature_extractor = Model(inputs=model.inputs, outputs=[layer.output for layer in model.layers]) # gives cumalative output of the input through this and previous layers.

        self.basep = basep
        self.bits = bits
        self.pvector = mock_image
        self.pshape = self.pvector.shape
        self.filters = init_kernel
        self.filters_shape = self.filters.shape
        self.strides = strides
        self.nFilters = nFilters
        self.mode = mode
        result = self.run_whetstone_to_fugu_utility(model)

        expected_num_spikes = (feature_extractor(mock_image).numpy() > 0.5).sum()
        calculated_num_spikes = len(result.index)
        assert expected_num_spikes == calculated_num_spikes

    def test_explicit_whetstone_2_fugu_conv2d_layer_multichannel_multifilter_mode_valid_strides_12(self):
        '''
            mode="valid", strides=(1,2). Answers are with biases equal to zero.

            Image:
                   Channel 1                Channel 2
                [[ 1.  2.  3.  4.  5.] , [[26. 27. 28. 29. 30.] 
                 [ 6.  7.  8.  9. 10.] ,  [31. 32. 33. 34. 35.] 
                 [11. 12. 13. 14. 15.] ,  [36. 37. 38. 39. 40.] 
                 [16. 17. 18. 19. 20.] ,  [41. 42. 43. 44. 45.] 
                 [21. 22. 23. 24. 25.]],  [46. 47. 48. 49. 50.]]

            Filters:
              Filter 1
                Channel 1   Channel 2
                [[ 1,  2],  [[5, 6],
                 [ 3,  4]]   [7, 8]]

              Filter 2
                Channel 1   Channel 2
                [[ 9, 10],  [[13, 14],
                 [11, 12]]   [15, 16]]

              Filter 3
                Channel 1   Channel 2
                [[17, 18],  [[21, 22],
                 [19, 20]]   [23, 24]]

            Convolution Answer:
               Filter 1 Out                Filter 2 Out                 Filter 3 Out
            [[ 772.  844.]              [[1828. 2028.]               [[2884. 3212.] 
             [ 952. 1024.]               [2328. 2528.]                [3704. 4032.] 
             [1132. 1204.]               [2828. 3028.]                [4524. 4852.] 
             [1312. 1384.]],             [3328. 3528.]],              [5344. 5672.]]
        '''
        from tensorflow.keras import Model, initializers
        image_height, image_width = 5, 5
        kernel_height, kernel_width = 2, 2
        nChannels = 2
        nFilters = 3
        input_shape = (image_height,image_width,nChannels)
        strides = (1,2)
        init_kernel = generate_keras_kernel(kernel_height,kernel_width,nFilters,nChannels)
        init_bias = np.array([-320., -653., -786.]).reshape((nFilters,)) # Results in 1 spike from filter 1, 2 spikes from filter 2, 2 spikes from filter 3. Spikes occur when filter outputs are > 0.5.
        kernel_initializer = initializers.constant(np.flip(init_kernel,(0,1))) # [METHOD 2] keras doesn't flip the filter during the convolution; so force the array flip manually.
        bias_initializer = initializers.constant(init_bias)
        mock_image = generate_mock_image(image_height,image_width,nChannels).astype(float)
        mode = "valid"
        basep = 3
        bits = 4

        model = Sequential()
        model.add(Conv2D(nFilters, (kernel_height, kernel_width), strides=strides, padding=mode, activation=None, use_bias=True, input_shape=input_shape, name="one", kernel_initializer=kernel_initializer, bias_initializer=bias_initializer))
        feature_extractor = Model(inputs=model.inputs, outputs=[layer.output for layer in model.layers]) # gives cumalative output of the input through this and previous layers.

        self.basep = basep
        self.bits = bits
        self.pvector = mock_image
        self.pshape = self.pvector.shape
        self.filters = init_kernel
        self.filters_shape = self.filters.shape
        self.strides = strides
        self.nFilters = nFilters
        self.mode = mode
        result = self.run_whetstone_to_fugu_utility(model)

        expected_num_spikes = (feature_extractor(mock_image).numpy() > 0.5).sum()
        calculated_num_spikes = len(result.index)
        assert expected_num_spikes == calculated_num_spikes

    def test_explicit_whetstone_2_fugu_conv2d_layer_multichannel_multifilter_mode_valid_strides_21(self):
        '''
            mode="valid", strides=(2,1). Answers are with biases equal to zero.

            Image:
                   Channel 1                Channel 2
                [[ 1.  2.  3.  4.  5.] , [[26. 27. 28. 29. 30.] 
                 [ 6.  7.  8.  9. 10.] ,  [31. 32. 33. 34. 35.] 
                 [11. 12. 13. 14. 15.] ,  [36. 37. 38. 39. 40.] 
                 [16. 17. 18. 19. 20.] ,  [41. 42. 43. 44. 45.] 
                 [21. 22. 23. 24. 25.]],  [46. 47. 48. 49. 50.]]

            Filters:
              Filter 1
                Channel 1   Channel 2
                [[ 1,  2],  [[5, 6],
                 [ 3,  4]]   [7, 8]]

              Filter 2
                Channel 1   Channel 2
                [[ 9, 10],  [[13, 14],
                 [11, 12]]   [15, 16]]

              Filter 3
                Channel 1   Channel 2
                [[17, 18],  [[21, 22],
                 [19, 20]]   [23, 24]]

            Convolution Answer:
               Filter 1 Out                Filter 2 Out                 Filter 3 Out
            [[ 772.  808.  844.  880.]   [[1828. 1928. 2028. 2128.]   [[2884. 3048. 3212. 3376.] 
             [1132. 1168. 1204. 1240.]],  [2828. 2928. 3028. 3128.]],  [4524. 4688. 4852. 5016.]]
        '''
        from tensorflow.keras import Model, initializers
        image_height, image_width = 5, 5
        kernel_height, kernel_width = 2, 2
        nChannels = 2
        nFilters = 3
        input_shape = (image_height,image_width,nChannels)
        strides = (2,1)
        init_kernel = generate_keras_kernel(kernel_height,kernel_width,nFilters,nChannels)
        init_bias = np.array([-320., -653., -786.]).reshape((nFilters,)) # Results in 1 spike from filter 1, 2 spikes from filter 2, 2 spikes from filter 3. Spikes occur when filter outputs are > 0.5.
        kernel_initializer = initializers.constant(np.flip(init_kernel,(0,1))) # [METHOD 2] keras doesn't flip the filter during the convolution; so force the array flip manually.
        bias_initializer = initializers.constant(init_bias)
        mock_image = generate_mock_image(image_height,image_width,nChannels).astype(float)
        mode = "valid"
        basep = 3
        bits = 4

        model = Sequential()
        model.add(Conv2D(nFilters, (kernel_height, kernel_width), strides=strides, padding=mode, activation=None, use_bias=True, input_shape=input_shape, name="one", kernel_initializer=kernel_initializer, bias_initializer=bias_initializer))
        feature_extractor = Model(inputs=model.inputs, outputs=[layer.output for layer in model.layers]) # gives cumalative output of the input through this and previous layers.

        self.basep = basep
        self.bits = bits
        self.pvector = mock_image
        self.pshape = self.pvector.shape
        self.filters = init_kernel
        self.filters_shape = self.filters.shape
        self.strides = strides
        self.nFilters = nFilters
        self.mode = mode
        result = self.run_whetstone_to_fugu_utility(model)

        expected_num_spikes = (feature_extractor(mock_image).numpy() > 0.5).sum()
        calculated_num_spikes = len(result.index)
        assert expected_num_spikes == calculated_num_spikes

    def test_explicit_whetstone_2_fugu_conv2d_layer_multichannel_multifilter_mode_valid_strides_22(self):
        '''
            mode="valid", strides=(2,2). Answers are with biases equal to zero.

            Image:
                   Channel 1                Channel 2
                [[ 1.  2.  3.  4.  5.] , [[26. 27. 28. 29. 30.] 
                 [ 6.  7.  8.  9. 10.] ,  [31. 32. 33. 34. 35.] 
                 [11. 12. 13. 14. 15.] ,  [36. 37. 38. 39. 40.] 
                 [16. 17. 18. 19. 20.] ,  [41. 42. 43. 44. 45.] 
                 [21. 22. 23. 24. 25.]],  [46. 47. 48. 49. 50.]]

            Filters:
              Filter 1
                Channel 1   Channel 2
                [[ 1,  2],  [[5, 6],
                 [ 3,  4]]   [7, 8]]

              Filter 2
                Channel 1   Channel 2
                [[ 9, 10],  [[13, 14],
                 [11, 12]]   [15, 16]]

              Filter 3
                Channel 1   Channel 2
                [[17, 18],  [[21, 22],
                 [19, 20]]   [23, 24]]

            Convolution Answer:
               Filter 1 Out                Filter 2 Out                 Filter 3 Out
            [[ 772.  844.]               [[1828. 2028.]               [[2884. 3212.] 
             [1132. 1204.]],              [2828. 3028.]],              [4524. 4852.]]
        '''
        from tensorflow.keras import Model, initializers
        image_height, image_width = 5, 5
        kernel_height, kernel_width = 2, 2
        nChannels = 2
        nFilters = 3
        input_shape = (image_height,image_width,nChannels)
        strides = (2,2)
        init_kernel = generate_keras_kernel(kernel_height,kernel_width,nFilters,nChannels)
        init_bias = np.array([-320., -653., -786.]).reshape((nFilters,)) # Results in 1 spike from filter 1, 2 spikes from filter 2, 2 spikes from filter 3. Spikes occur when filter outputs are > 0.5.
        kernel_initializer = initializers.constant(np.flip(init_kernel,(0,1))) # [METHOD 2] keras doesn't flip the filter during the convolution; so force the array flip manually.
        bias_initializer = initializers.constant(init_bias)
        mock_image = generate_mock_image(image_height,image_width,nChannels).astype(float)
        mode = "valid"
        basep = 3
        bits = 4

        model = Sequential()
        model.add(Conv2D(nFilters, (kernel_height, kernel_width), strides=strides, padding=mode, activation=None, use_bias=True, input_shape=input_shape, name="one", kernel_initializer=kernel_initializer, bias_initializer=bias_initializer))
        feature_extractor = Model(inputs=model.inputs, outputs=[layer.output for layer in model.layers]) # gives cumalative output of the input through this and previous layers.

        self.basep = basep
        self.bits = bits
        self.pvector = mock_image
        self.pshape = self.pvector.shape
        self.filters = init_kernel
        self.filters_shape = self.filters.shape
        self.strides = strides
        self.nFilters = nFilters
        self.mode = mode
        result = self.run_whetstone_to_fugu_utility(model)

        expected_num_spikes = (feature_extractor(mock_image).numpy() > 0.5).sum()
        calculated_num_spikes = len(result.index)
        assert expected_num_spikes == calculated_num_spikes

    @pytest.mark.parametrize("mode",["same","valid"])
    def test_whetstone_2_fugu_conv2d_layer_with_batchnormalization(self,mode):
        '''
            TODO: Add test information here
        '''
        from tensorflow.keras import Model, initializers
        image_height, image_width = 3, 3
        kernel_height, kernel_width = 2, 2
        nChannels = 2
        nFilters = 3
        input_shape = (image_height,image_width,nChannels)
        strides = (1,1)
        init_kernel = generate_keras_kernel(kernel_height,kernel_width,nFilters,nChannels)
        init_bias = np.array([-320., -653., -786.]).reshape((nFilters,))
        kernel_initializer = initializers.constant(np.flip(init_kernel,(0,1))) # [METHOD 2] keras doesn't flip the filter during the convolution; so force the array flip manually.
        bias_initializer = initializers.constant(init_bias)
        mock_image = generate_mock_image(image_height,image_width,nChannels).astype(float)
        basep = 3
        bits = 4

        gamma_initializer = initializers.constant(2.)
        beta_initializer = initializers.constant(3.)
        moving_mean_initializer = initializers.constant(4.)
        moving_variance_initializer = initializers.constant(5.)

        model = Sequential()
        model.add(Conv2D(nFilters, (kernel_height, kernel_width), strides=strides, padding=mode, activation=None, use_bias=True, input_shape=input_shape, name="one", kernel_initializer=kernel_initializer, bias_initializer=bias_initializer))
        model.add(BatchNormalization(beta_initializer=beta_initializer, gamma_initializer=gamma_initializer, moving_mean_initializer=moving_mean_initializer, moving_variance_initializer=moving_variance_initializer))
        feature_extractor = Model(inputs=model.inputs, outputs=[layer.output for layer in model.layers]) # gives cumalative output of the input through this and previous layers.

        conv2d_layer = model.layers[0]
        bnorm_layer = model.layers[1]
        new_weights, new_biases = merge_layers(conv2d_layer,bnorm_layer)
        new_kernel_initializer = initializers.constant(new_weights)
        new_biases_initializer = initializers.constant(new_biases)

        merged_model = Sequential()
        merged_model.add(Conv2D(nFilters, (kernel_height, kernel_width), strides=strides, padding=mode, activation=None, use_bias=True, input_shape=input_shape, name="merged", kernel_initializer=new_kernel_initializer, bias_initializer=new_biases_initializer))
        merged_calculated = merged_model.layers[0](mock_image)[0,:,:,:].numpy()

        self.basep = basep
        self.bits = bits
        self.pvector = mock_image
        self.pshape = self.pvector.shape
        self.filters = init_kernel
        self.filters_shape = self.filters.shape
        self.strides = strides
        self.nFilters = nFilters
        self.mode = mode
        result = self.run_whetstone_to_fugu_utility(model)

        expected_spike_count = (feature_extractor(mock_image)[1].numpy() > 0.5).sum().astype(int)
        calculated_spike_count = len(result.index)
        assert expected_spike_count == calculated_spike_count

    @pytest.mark.parametrize("strides",[(1,1),(1,2),(2,1),(2,2)])
    @pytest.mark.parametrize("mode", ["same", "valid"])
    @pytest.mark.parametrize("nChannels", [1,2,3])
    @pytest.mark.parametrize("nFilters", [1,2,3])
    def test_batch_normalization_removal(self,strides,mode,nChannels,nFilters):
        '''
            TODO: Add test information here
        '''
        from tensorflow.keras import Model, initializers

        image_height, image_width = 3, 3
        kernel_height, kernel_width = 2, 2
        input_shape = (image_height,image_width,nChannels)
        init_kernel = generate_keras_kernel(kernel_height,kernel_width,nFilters,nChannels)
        init_bias = -53*np.ones((nFilters,))
        kernel_initializer = initializers.constant(np.flip(init_kernel,(0,1))) # [METHOD 2] keras doesn't flip the filter during the convolution; so force the array flip manually.
        bias_initializer = initializers.constant(init_bias)
        mock_image = generate_mock_image(image_height,image_width,nChannels).astype(float)
        nSpikes = 2
        
        gamma_initializer = initializers.constant(2.)
        beta_initializer = initializers.constant(3.)
        moving_mean_initializer = initializers.constant(4.)
        moving_variance_initializer = initializers.constant(5.)

        model = Sequential()
        model.add(Conv2D(nFilters, (kernel_height, kernel_width), strides=strides, padding=mode, activation=None, use_bias=True, input_shape=input_shape, name="one", kernel_initializer=kernel_initializer, bias_initializer=bias_initializer))
        model.add(BatchNormalization(beta_initializer=beta_initializer, gamma_initializer=gamma_initializer, moving_mean_initializer=moving_mean_initializer, moving_variance_initializer=moving_variance_initializer))
        feature_extractor = Model(inputs=model.inputs, outputs=[layer.output for layer in model.layers]) # gives cumalative output of the input through this and previous layers.
        conv2d_layer = model.layers[0]
        bnorm_layer = model.layers[1]
        conv2d_result = feature_extractor(mock_image)[0][0,:,:,:].numpy()
        expected = feature_extractor(mock_image)[1][0,:,:,:].numpy()
        calculated = normalization(conv2d_result,bnorm_layer)
        assert np.allclose(calculated,expected,rtol=1e-6,atol=1e-5)

        new_weights, new_biases = merge_layers(conv2d_layer,bnorm_layer)
        new_kernel_initializer = initializers.constant(new_weights)
        new_biases_initializer = initializers.constant(new_biases)

        new_model = Sequential()
        new_model.add(Conv2D(nFilters, (kernel_height, kernel_width), strides=strides, padding=mode, activation=None, use_bias=True, input_shape=input_shape, name="merged", kernel_initializer=new_kernel_initializer, bias_initializer=new_biases_initializer))
        calculated = new_model.layers[0](mock_image)[0,:,:,:].numpy()
        assert np.allclose(calculated,expected,rtol=1e-6,atol=1e-5)

    def run_whetstone_to_fugu_utility(self, keras_model):
        scaffold = Scaffold()
        scaffold.add_brick(BaseP_Input(self.pvector,p=self.basep,bits=self.bits,collapse_binary=False,name="I",time_dimension=False),"input")
        scaffold = whetstone_2_fugu(keras_model,self.basep,self.bits,scaffold=scaffold)
        scaffold.lay_bricks()
        scaffold.summary(verbose=1)
        backend = snn_Backend()
        backend_args = {}
        backend.compile(scaffold, backend_args)
        result = backend.run(10)
        return result

    # Auxillary/helper functions
    def get_num_output_neurons(self, thresholds):
        input_shape = np.array(self.pshape)[1:3]
        kernel_shape = np.array(self.filters.shape)[:2]
        full_output_shape = input_shape + kernel_shape - 1

        #TODO: Do I need this conditional statement here?
        if not hasattr(thresholds, "__len__") and (not isinstance(thresholds, str)):
            if self.mode == "full":
                thresholds_size = full_output_shape.prod()

            if self.mode == "valid":
                lmins = np.minimum(input_shape, kernel_shape)
                lb = lmins - 1
                ub = np.array(full_output_shape) - lmins
                thresholds_size = (ub[0] - lb[0] + 1) * (ub[1] - lb[1] + 1)

            if self.mode == "same":
                lb = np.floor(0.5 * (full_output_shape - input_shape))
                ub = np.floor(0.5 * (full_output_shape + input_shape) - 1)
                thresholds_size = (ub[0] - lb[0] + 1) * (ub[1] - lb[1] + 1)
        else:
            if thresholds.ndim == 2:
                thresholds_size = np.size(thresholds)
            elif thresholds.ndim ==4:
                thresholds_size = np.size(thresholds[0,:,:,0])

        return thresholds_size * self.nFilters

    def output_spike_positions(self, basep, bits, pvector, filters, thresholds):
        thresholds_size = self.get_num_output_neurons(thresholds)
        offset = 4  # begin/complete nodes for input and output nodes
        input_basep_len = np.size(pvector) * basep * bits
        output_ini_position = offset + input_basep_len
        output_end_position = offset + input_basep_len + thresholds_size
        return [output_ini_position, output_end_position]

    def output_mask(self, output_positions, result):
        ini = output_positions[0]
        end = output_positions[1]
        return (result["neuron_number"] >= ini) & (result["neuron_number"] <= end)

    def calculated_spikes(self,thresholds,result):
        # get output positions in result
        output_positions = self.output_spike_positions(
            self.basep, self.bits, self.pvector, self.filters, thresholds
        )
        output_mask = self.output_mask(output_positions, result)

        calculated_spikes = list(result[output_mask].to_numpy()[:, 0])
        return calculated_spikes

    def expected_spikes(self,nSpikes):
        # Check calculations
        if nSpikes == 0:
            expected_spikes = list(np.array([]))
        else:
            expected_spikes = list(np.ones((nSpikes,)))

        return expected_spikes

# Auxillary/Helper functions
def normalization(batch, bnorm_layer):
    gamma, beta, mean, variance = bnorm_layer.get_weights()
    epsilon = bnorm_layer.epsilon
    return apply_normalization(batch,gamma,beta,mean,variance,epsilon)

def apply_normalization(batch,gamma, beta, moving_mean, moving_var, epsilon):
    return gamma*(batch - moving_mean) / np.sqrt(moving_var+epsilon) + beta

def merge_layers(conv2d_layer, bnorm_layer):
    gamma, beta, mean, variance = bnorm_layer.get_weights()
    epsilon = bnorm_layer.epsilon
    weights = conv2d_layer.get_weights()[0]
    biases = conv2d_layer.get_weights()[1]

    stdev = np.sqrt(variance + epsilon)
    new_weights = weights * gamma / stdev
    new_biases = (gamma / stdev) * (biases - mean) + beta
    return new_weights, new_biases
