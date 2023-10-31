import numpy as np
import pytest
from scipy.signal import convolve2d

from fugu.backends import snn_Backend
from fugu.bricks.keras_convolution_bricks import keras_convolution_2d as convolution_2d
from fugu.bricks.dense_bricks import dense_layer_1d, dense_layer_2d
from fugu.bricks.input_bricks import BaseP_Input, Vector_Input
from fugu.bricks.pooling_bricks import pooling_1d, pooling_2d
from fugu.scaffold import Scaffold
from fugu.utils.whetstone import whetstone_2_fugu

import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from whetstone.layers import Spiking_BRelu
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

    def test_whetstone_2_fugu_conv2d_layer(self):
        '''
            [[1,2,3]                 [[ 4,11,18, 9]
             [4,5,6]  *  [[1,2]  =    [18,37,47,21]
             [7,8,9]]     [3,4]]      [36,67,77,33]
                                      [14,23,26, 9]
            
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
                                      [21,52,59,36]
        '''
        from tensorflow.keras import Model, initializers
        input_shape = (3,3,1)
        strides = (1,1)
        init_kernel = np.reshape(np.flip(np.arange(1,5)),(1,2,2,1)) # [METHOD 2] keras doesn't flip the filter during the convolution; so force the array flip manually.
        init_bias = np.flip(np.array([[23,33,24],[53,63,42],[52,59,36]])).reshape(1,3,3,1)
        init_bias = -52.6*np.ones((1,))
        kernel_initializer = initializers.constant(init_kernel)
        bias_initializer = initializers.constant(init_bias)
        nSpikes = 2

        model = Sequential()
        model.add(Conv2D(1, (2, 2), strides=strides, padding='same', activation=None, use_bias=True, input_shape=input_shape, name="one", kernel_initializer=kernel_initializer, bias_initializer=bias_initializer))
        # model.add(Conv2D(1, (2, 2), strides=strides, padding='same', activation=None, use_bias=True, input_shape=input_shape, name="two", kernel_initializer=kernel_initializer, bias_initializer=bias_initializer))
        model.add(Spiking_BRelu(input_shape=(3,3),sharpness=1.0,name="spike"))
        thresholds = model.layers[0].get_weights()[0].reshape(2,2)
        mock_image = np.arange(1,10,dtype=float).reshape(1,3,3,1)
        calculated = model.layers[0](mock_image)[0,:,:,0].numpy().tolist() # gives the output of the input through this layer only
        feature_extractor = Model(inputs=model.inputs, outputs=[layer.output for layer in model.layers]) # gives cumalative output of the input through this and previous layers.

        self.basep = 3
        self.bits = 3
        self.pvector = mock_image.reshape(3,3)
        self.pshape = self.pvector.shape
        self.filters = np.flip(init_kernel.reshape(2,2))
        self.filters_shape = self.filters.shape
        self.strides = strides
        thresholds = calculated
        scaffold = Scaffold()
        scaffold.add_brick(BaseP_Input(mock_image.reshape(3,3),p=self.basep,bits=self.bits,collapse_binary=False,name="I",time_dimension=False),"input")
        scaffold = whetstone_2_fugu(model,self.basep,self.bits,scaffold=scaffold)
        scaffold.lay_bricks()

        # Override output neuron threshold values in scaffold
        output_neuron_names = np.array(scaffold.graph.nodes)[-9:]
        for k, name in enumerate(output_neuron_names):
            print(f"threshold: {scaffold.graph.nodes[name]['threshold']}  Spiking_BRelu: {feature_extractor(mock_image)[1][0,:,:,0].numpy().flatten()[k]}")
        
        keras_convolution_answer = feature_extractor(mock_image)[0][0,:,:,0].numpy()
        keras_spike_answer = feature_extractor(mock_image)[1][0,:,:,0].numpy()
        new_fugu_thresholds = (keras_convolution_answer - init_bias).astype(int) - 0.1*keras_spike_answer

        # Assign new thresholds values to convolution output neurons in the scaffold
        for k, name in enumerate(output_neuron_names):
            scaffold.graph.nodes[name]['threshold'] = new_fugu_thresholds.flatten()[k]

        scaffold.summary(verbose=1)
        backend = snn_Backend()
        backend_args = {}
        backend.compile(scaffold, backend_args)
        result = backend.run(5)

        assert self.expected_spikes(nSpikes) == self.calculated_spikes(new_fugu_thresholds, result)
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
class Test_Whetstone_2_Fugu_Normalization_On:
    @pytest.mark.xfail(reason="Not implemented.")
    def test_layers(self):
        assert False

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

class Test_Whetstone_2_Fugu_DenseLayer:
    pass

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

    @pytest.mark.parametrize("strides,expected", [((1,1), [[1.,3.],[4.,10.]]), ((1,2), [[3.],[10.]]), ((2,1), [[4.,10.]]), ((2,2), [[10.]])])
    def test_simple_conv2d_strides(self, strides, expected):
        '''
        *** IMPORTANT ***
        Keras returns a "flipped" convolution result. That is, using scipy.signal.convolve2d (or hand calculated results) for image=[[1,1],[1,1]] and kernel=[[1,2],[3,4]] with mode="same" and
        strides=(1,1) will produce a 2D convolution result equal to [[1,3],[4,10]]. Whereas Keras returns [[10,4],[3,1]] which is flipped version (both rows & columns) of the "normal" result. Consquently,
        this behavior is important to understand when using non-unit strides in a Convolution2D layer in Keras because the answer that is return will coincide with the "flipped" result. For example, using
        the same example image and kernel provided early with mode/padding="same" and strides=(2,2) will return [[10]] in Keras but the "normal" calculation will return [[1]]. The reason for this behavior due
        to the flipped result that contains the "10" in the top left entry when strides=(1,1). 
        '''
        from tensorflow.keras import Model, initializers
        
        input_shape = (2,2,1)
        init_kernel = np.reshape(np.arange(1,5),(1,2,2,1))
        init_bias = np.zeros((1,))
        kernel_initializer = initializers.constant(init_kernel)
        bias_initializer = initializers.constant(init_bias)

        model = Sequential()
        model.add(Conv2D(1, (2, 2), strides=strides, padding='same', activation=None, use_bias=True, input_shape=input_shape, name="one", kernel_initializer=kernel_initializer, bias_initializer=bias_initializer))
        # feature_extractor = Model(inputs=model.inputs, outputs=model.get_layer(name="one").output)
        feature_extractor = Model(inputs=model.inputs, outputs=[layer.output for layer in model.layers])

        image = np.ones((1,2,2,1))
        calculated = np.flip(feature_extractor(image)[0,:,:,0].numpy()).tolist()
        assert expected == calculated

    @pytest.mark.xfail(reason="Not implemented.")
    def test_simple_conv2d_strides_tmp(self):
        from tensorflow.keras import Model, initializers
        
        input_shape = (2,2,1)
        init_kernel = np.reshape(np.arange(1,5),(1,2,2,1))
        init_bias = np.zeros((1,))
        kernel_initializer = initializers.constant(init_kernel)
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

        image = np.ones((1,2,2,1))
        calculated = np.flip(feature_extractor11(image)[0,:,:,0].numpy()).tolist()
        expected = [[1.,3.],[4.,10.]]
        assert expected == calculated
        assert False

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

class Test_Keras_MaxPooling2d:
    @pytest.mark.parametrize("mock_input,pool_size,pool_strides,expected", [(np.arange(1,10).reshape((1,3,3,1)), (1,1), (1,1), [[1.,2.,3.],[4.,5.,6.],[7.,8.,9.]]),
                                                                            (np.arange(1,10).reshape((1,3,3,1)), (2,2), (1,1), [[5.,6.,6.],[8.,9.,9.],[8.,9.,9.]])])    
    def test_simple_max_pooling2d(self,mock_input,pool_size,pool_strides,expected):
        from tensorflow.keras import Model, initializers
        input_shape = mock_input[0].shape

        model = Sequential()
        model.add(MaxPooling2D(pool_size=pool_size, strides=pool_strides, padding='same', name="two", input_shape=input_shape))

        calculated = model.layers[0](mock_input)[0,:,:,0].numpy().tolist()
        assert expected == calculated

class Test_Keras_Multilayer_Model:
    @pytest.mark.parametrize("mock_conv2d_output,pool_size,pool_strides,expected", [(np.arange(1,10).reshape((1,3,3,1)), (1,1), (1,1), [[1.,3.],[4.,10.]]),
                                                                                    (np.arange(1,10).reshape((1,3,3,1)), (2,2), (1,1), [[1.,3.],[4.,10.]])])
    @pytest.mark.xfail(reason="Not implemented.")
    def test_simple_multilayer_model(self, mock_conv2d_output, pool_size, pool_strides, expected):
        from tensorflow.keras import Model, initializers
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