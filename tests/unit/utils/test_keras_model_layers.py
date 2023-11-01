import numpy as np
import pytest

import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras import Model, initializers

class Test_Keras_Conv2d:
    def test_simple_conv2d(self):
        
        input_shape = (2,2,1)
        init_kernel = np.reshape(np.arange(1,5),(1,2,2,1))
        init_bias = np.zeros((1,))
        kernel_initializer = initializers.constant(np.flip(init_kernel))
        bias_initializer = initializers.constant(init_bias)

        model = Sequential()
        model.add(Conv2D(1, (2, 2), padding='same', activation=None, use_bias=True, input_shape=input_shape, name="one", kernel_initializer=kernel_initializer, bias_initializer=bias_initializer))

        # feature_extractor = Model(inputs=model.inputs, outputs=model.get_layer(name="one").output)
        feature_extractor = Model(inputs=model.inputs, outputs=[layer.output for layer in model.layers])
        image = np.arange(1,5).reshape(1,2,2,1)
        calculated = feature_extractor(image)[0,:,:,0].numpy().tolist()
        expected = [[20.,16.],[24.,16.]]
        assert expected == calculated

    @pytest.mark.parametrize("strides,expected", [((1,1), [[20.,16.],[24.,16.]]), ((1,2), [[20.],[24.]]), ((2,1), [[20.,16.]]), ((2,2), [[20.]])])
    def test_simple_conv2d_strides(self, strides, expected):
        '''
        *** IMPORTANT ***
        Keras does NOT "flip" the kernel before applying the convolution operator "*" to the input image. That is, using scipy.signal.convolve2d (or hand calculated results) for
        image=[[1,2],[3,4]] and kernel=[[1,2],[3,4]] with mode="full" and strides=(1,1) will produce a 2D convolution result equal to [[1,4,4],[6,20,16],[9,24,16]].

        To reproduce the Keras padding="same" result, you must
            (1) "flip" (np.flip(...)) the input kernel matrix passed to the scipy.signal.convolve2d function and then
            (2) take the [1:,1:] sliced convolution result.

        Passing image=[[1,2],[3,4]] and the flipped (flipped both row and column) kernel=[[1,2],[3,4]] ([[4,3],[2,1]]) to scipy.signal.convolve2d with mode="full" returns a
        2D convolution result equal to [[4,11,6],[14,30,14],[6,11,4]]. The Keras' 2D convolution result using padding="same", image=[[1,2],[3,4]], and kernel=[[1,2],[3,4]]
        (i.e., same parameters, except kernel is NOT flipped when passed to Keras; Keras' doesn't have padding="full") results in a 2D convolution equation to [[30,14],[11,4]].
        '''
        
        input_shape = (2,2,1)
        init_kernel = np.reshape(np.arange(1,5),(1,2,2,1))
        init_bias = np.zeros((1,))
        kernel_initializer = initializers.constant(np.flip(init_kernel))
        bias_initializer = initializers.constant(init_bias)

        model = Sequential()
        model.add(Conv2D(1, (2, 2), strides=strides, padding='same', activation=None, use_bias=True, input_shape=input_shape, name="one", kernel_initializer=kernel_initializer, bias_initializer=bias_initializer))
        # feature_extractor = Model(inputs=model.inputs, outputs=model.get_layer(name="one").output)
        feature_extractor = Model(inputs=model.inputs, outputs=[layer.output for layer in model.layers])

        image = np.arange(1,5).reshape(1,2,2,1)
        calculated = feature_extractor(image)[0,:,:,0].numpy().tolist()
        assert expected == calculated

    @pytest.mark.xfail(reason="Not implemented.")
    def test_simple_conv2d_strides_tmp(self):
        #TODO: finish test using image.np.arange(1,10).reshape(1,3,3,1)
        
        input_shape = (3,3,1)
        init_kernel = np.reshape(np.arange(1,5),(1,2,2,1))
        init_bias = np.zeros((1,))
        kernel_initializer = initializers.constant(np.flip(init_kernel))
        bias_initializer = initializers.constant(init_bias)

        model11 = Sequential()
        model11.add(Conv2D(1, (2, 2), strides=(1,1), padding='valid', activation=None, use_bias=True, input_shape=input_shape, name="one", kernel_initializer=kernel_initializer, bias_initializer=bias_initializer))
        feature_extractor11 = Model(inputs=model11.inputs, outputs=model11.get_layer(name="one").output)
        # feature_extractor11 = Model(inputs=model11.inputs, outputs=[layer.output for layer in model11.layers])

        model12 = Sequential()
        model12.add(Conv2D(1, (2, 2), strides=(1,2), padding='valid', activation=None, use_bias=True, input_shape=input_shape, name="one", kernel_initializer=kernel_initializer, bias_initializer=bias_initializer))
        feature_extractor12 = Model(inputs=model12.inputs, outputs=model12.get_layer(name="one").output)
        # feature_extractor12 = Model(inputs=model12.inputs, outputs=[layer.output for layer in model12.layers])

        model21 = Sequential()
        model21.add(Conv2D(1, (2, 2), strides=(2,1), padding='valid', activation=None, use_bias=True, input_shape=input_shape, name="one", kernel_initializer=kernel_initializer, bias_initializer=bias_initializer))
        feature_extractor21 = Model(inputs=model21.inputs, outputs=model21.get_layer(name="one").output)
        # feature_extractor21 = Model(inputs=model21.inputs, outputs=[layer.output for layer in model21.layers])

        model22 = Sequential()
        model22.add(Conv2D(1, (2, 2), strides=(2,2), padding='valid', activation=None, use_bias=True, input_shape=input_shape, name="one", kernel_initializer=kernel_initializer, bias_initializer=bias_initializer))
        feature_extractor22 = Model(inputs=model22.inputs, outputs=model22.get_layer(name="one").output)
        # feature_extractor22 = Model(inputs=model22.inputs, outputs=[layer.output for layer in model22.layers])

        # image = np.ones((1,3,3,1))
        image = np.arange(1,10,dtype=float).reshape(1,3,3,1)
        calculated = feature_extractor11(image)[0,:,:,0].numpy().tolist()
        expected = [[1.,3.],[4.,10.]]
        assert expected == calculated
        assert False

    def test_multichannel_conv2d(self):
        #TODO: Update using "image = np.repeat(np.arange(1,5),nChannels).reshape(1,2,2,nChannels)" instead of "image=np.ones((1,2,2,nChannels))"
        
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
        #TODO: Update using "image = np.arange(1,5).reshape(1,2,2,1)" instead of "image=np.ones((1,2,2,1))"
        
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
        #TODO: Update using "image = np.repeat(np.arange(1,5),nChannels).reshape(1,2,2,nChannels)" instead of "image=np.ones((1,2,2,nChannels))"
        
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

class Test_Keras_MaxPooling2d:
    @pytest.mark.parametrize("mock_input,pool_size,pool_strides,expected", [(np.arange(1,10).reshape((1,3,3,1)), (1,1), (1,1), [[1.,2.,3.],[4.,5.,6.],[7.,8.,9.]]),
                                                                            (np.arange(1,10).reshape((1,3,3,1)), (2,2), (1,1), [[5.,6.,6.],[8.,9.,9.],[8.,9.,9.]])])    
    def test_simple_max_pooling2d(self,mock_input,pool_size,pool_strides,expected):
        input_shape = mock_input[0].shape

        model = Sequential()
        model.add(MaxPooling2D(pool_size=pool_size, strides=pool_strides, padding='same', name="two", input_shape=input_shape))

        calculated = model.layers[0](mock_input)[0,:,:,0].numpy().tolist()
        assert expected == calculated

class Test_Keras_Multilayer_Model:
    #TODO: Update using "image = np.arange(1,5).reshape(1,2,2,1)" instead of "image=np.ones((1,2,2,1))"
    @pytest.mark.parametrize("mock_conv2d_output,pool_size,pool_strides,expected", [(np.arange(1,10).reshape((1,3,3,1)), (1,1), (1,1), [[1.,3.],[4.,10.]]),
                                                                                    (np.arange(1,10).reshape((1,3,3,1)), (2,2), (1,1), [[1.,3.],[4.,10.]])])
    @pytest.mark.xfail(reason="Not implemented.")
    def test_simple_multilayer_model(self, mock_conv2d_output, pool_size, pool_strides, expected):
        input_shape = (2,2,1)
        init_kernel = np.reshape(np.arange(1,5),(1,2,2,1))
        init_bias = np.zeros((1,))
        kernel_initializer = initializers.constant(init_kernel)
        bias_initializer = initializers.constant(init_bias)

        model = Sequential()
        model.add(Conv2D(1, (2, 2), strides=(1,1), padding='same', activation=None, use_bias=True, input_shape=input_shape, name="one", kernel_initializer=kernel_initializer, bias_initializer=bias_initializer))
        model.add(MaxPooling2D(pool_size=pool_size, strides=pool_strides, padding='same', name="two"))
        # model.add(MaxPooling2D(pool_size=(2,2), strides=(1,1), padding='same', name="two"))

        feature_extractor = Model(inputs=model.inputs, outputs=model.get_layer(name="two").output)
        # feature_extractor = Model(inputs=model.inputs, outputs=[layer.output for layer in model.layers])
        image = np.ones((1,2,2,1))
        calculated = np.flip(feature_extractor(image)[0,:,:,0].numpy()).tolist()
        assert expected == calculated