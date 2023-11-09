# isort: skip_file
import numpy as np
import pytest
from scipy.signal import convolve2d

from fugu.backends import snn_Backend
from fugu.bricks.keras_convolution_bricks import keras_convolution_2d as convolution_2d
from fugu.bricks.dense_bricks import dense_layer_1d, dense_layer_2d
from fugu.bricks.input_bricks import BaseP_Input, Vector_Input
from fugu.bricks.pooling_bricks import pooling_1d, pooling_2d
from fugu.scaffold import Scaffold
from fugu.utils.keras_helpers import keras_convolve2d, keras_convolve2d_4dinput, generate_keras_kernel, generate_mock_image
from fugu.utils.whetstone import whetstone_2_fugu

import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, BatchNormalization
from whetstone.layers import Spiking_BRelu
from whetstone.utils import layer_utils

# Turn off black formatting for this file
# fmt: off

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
        input_shape = (image_height,image_width,nChannels)
        strides = (1,1)
        init_kernel = generate_keras_kernel(kernel_height,kernel_width,nFilters,nChannels)
        init_bias = bias*np.ones((nFilters,))
        kernel_initializer = initializers.constant(np.flip(init_kernel)) # [METHOD 2] keras doesn't flip the filter during the convolution; so force the array flip manually.
        bias_initializer = initializers.constant(init_bias)
        mode = "same"

        model = Sequential()
        model.add(Conv2D(nFilters, (kernel_height, kernel_width), strides=strides, padding=mode, activation=None, use_bias=True, input_shape=input_shape, name="one", kernel_initializer=kernel_initializer, bias_initializer=bias_initializer))
        model.add(Spiking_BRelu(input_shape=(3,3),sharpness=1.0,name="spike"))
        mock_image = generate_mock_image(image_height,image_width,nChannels).astype(float)
        calculated = model.layers[0](mock_image)[0,:,:,0].numpy().tolist() # gives the output of the input through this layer only
        feature_extractor = Model(inputs=model.inputs, outputs=[layer.output for layer in model.layers]) # gives cumalative output of the input through this and previous layers.

        self.basep = 3
        self.bits = 3
        self.pvector = mock_image.reshape(3,3)
        self.pshape = self.pvector.shape
        self.filters = init_kernel.reshape(2,2)
        self.filters_shape = self.filters.shape
        self.strides = strides
        scaffold = Scaffold()
        scaffold.add_brick(BaseP_Input(mock_image.reshape(image_height,image_width),p=self.basep,bits=self.bits,collapse_binary=False,name="I",time_dimension=False),"input")
        scaffold = whetstone_2_fugu(model,self.basep,self.bits,scaffold=scaffold)
        scaffold.lay_bricks()
        scaffold.summary(verbose=1)

        backend = snn_Backend()
        backend_args = {}
        backend.compile(scaffold, backend_args)
        result = backend.run(5)
        new_fugu_thresholds = 0.5*np.ones(feature_extractor(mock_image)[0][0,:,:,0].numpy().shape)
        assert self.expected_spikes(nSpikes) == self.calculated_spikes(new_fugu_thresholds, result)

    def test_explicit_whetstone_2_fugu_conv2d_layer(self):
        '''
            [[1,2,3]                 [[ 4,11,18, 9]
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

            [[1,2,3]                 [[ 1, 4, 7, 6]
            [4,5,6]  *  [[1,2]  =     [ 7,23,33,24]
            [7,8,9]]     [3,4]]       [19,53,63,42]
                                        [21,52,59,36]]
        '''
        from tensorflow.keras import Model, initializers
        input_shape = (3,3,1)
        strides = (1,1)
        init_kernel = np.reshape(np.arange(1,5),(1,2,2,1))
        init_bias = -52.6*np.ones((1,))
        kernel_initializer = initializers.constant(np.flip(init_kernel)) # [METHOD 2] keras doesn't flip the filter during the convolution; so force the array flip manually.
        bias_initializer = initializers.constant(init_bias)
        nSpikes = 2

        model = Sequential()
        model.add(Conv2D(1, (2, 2), strides=strides, padding='same', activation=None, use_bias=True, input_shape=input_shape, name="one", kernel_initializer=kernel_initializer, bias_initializer=bias_initializer))
        model.add(Spiking_BRelu(input_shape=(3,3),sharpness=1.0,name="spike"))
        mock_image = np.arange(1,10,dtype=float).reshape(1,3,3,1)
        calculated = model.layers[0](mock_image)[0,:,:,0].numpy().tolist() # gives the output of the input through this layer only
        feature_extractor = Model(inputs=model.inputs, outputs=[layer.output for layer in model.layers]) # gives cumalative output of the input through this and previous layers.

        self.basep = 3
        self.bits = 3
        self.pvector = mock_image.reshape(3,3)
        self.pshape = self.pvector.shape
        self.filters = init_kernel.reshape(2,2)
        self.filters_shape = self.filters.shape
        self.strides = strides
        scaffold = Scaffold()
        scaffold.add_brick(BaseP_Input(mock_image.reshape(3,3),p=self.basep,bits=self.bits,collapse_binary=False,name="I",time_dimension=False),"input")
        scaffold = whetstone_2_fugu(model,self.basep,self.bits,scaffold=scaffold)
        scaffold.lay_bricks()

        scaffold.summary(verbose=1)
        backend = snn_Backend()
        backend_args = {}
        backend.compile(scaffold, backend_args)
        result = backend.run(5)

        new_fugu_thresholds = 0.5*np.ones(feature_extractor(mock_image)[0][0,:,:,0].numpy().shape)
        assert self.expected_spikes(nSpikes) == self.calculated_spikes(new_fugu_thresholds, result)

    @pytest.mark.parametrize("strides",[(1,1),(1,2),(2,1),(2,2)])
    @pytest.mark.parametrize("mode", ["same", "valid"])
    @pytest.mark.parametrize("nChannels", [1,2,3])
    @pytest.mark.parametrize("nFilters", [1,2,3])
    def test_batch_normalization_removal(self,strides,mode,nChannels,nFilters):
        from tensorflow.keras import Model, initializers

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

        image_height, image_width = 3, 3
        kernel_height, kernel_width = 2, 2
        input_shape = (image_height,image_width,nChannels)
        init_kernel = generate_keras_kernel(kernel_height,kernel_width,nFilters,nChannels)
        init_bias = -53*np.ones((nFilters,))
        kernel_initializer = initializers.constant(np.flip(init_kernel)) # [METHOD 2] keras doesn't flip the filter during the convolution; so force the array flip manually.
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

    @pytest.mark.xfail(reason="Not implemented.")
    def test_4d_tensor_input_whetstone_2_fugu_conv2d_layer(self):
        assert False

    # Auxillary/helper functions
    def get_num_output_neurons(self, thresholds):
        Am, An = self.pshape
        Bm, Bn = self.filters_shape
        Gm, Gn = Am + Bm - 1, An + Bn - 1

        if not hasattr(thresholds, "__len__") and (not isinstance(thresholds, str)):
            if self.mode == "full":
                thresholds_size = (Gm) * (Gn)

            if self.mode == "valid":
                lmins = np.minimum((Am, An), (Bm, Bn))
                lb = lmins - 1
                ub = np.array([Gm, Gn]) - lmins
                thresholds_size = (ub[0] - lb[0] + 1) * (ub[1] - lb[1] + 1)

            if self.mode == "same":
                apos = np.array([Am, An])
                gpos = np.array([Gm, Gn])

                lb = np.floor(0.5 * (gpos - apos))
                ub = np.floor(0.5 * (gpos + apos) - 1)
                thresholds_size = (ub[0] - lb[0] + 1) * (ub[1] - lb[1] + 1)
        else:
            thresholds_size = np.size(thresholds)

        return thresholds_size

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
