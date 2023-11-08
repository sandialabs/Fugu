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
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, BatchNormalization
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

class Test_Whetstone_2_Fugu:
    @pytest.mark.skip(reason="Not implemented. And takes to long for 'xfail'.")
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
        input_shape = (3,3,1)
        strides = (1,1)
        init_kernel = np.reshape(np.arange(1,5),(1,2,2,1))
        init_bias = bias*np.ones((1,))
        kernel_initializer = initializers.constant(np.flip(init_kernel)) # [METHOD 2] keras doesn't flip the filter during the convolution; so force the array flip manually.
        bias_initializer = initializers.constant(init_bias)

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

        # Override output neuron threshold values in scaffold
        output_neuron_names = np.array(scaffold.graph.nodes)[-9:]
        
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

        # Override output neuron threshold values in scaffold
        output_neuron_names = np.array(scaffold.graph.nodes)[-9:]
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

    @pytest.mark.xfail(reason="Not implemented.")
    def test_batch_normalization_removal(self):
        from tensorflow.keras import Model, initializers
        image_height, image_width = 3, 3
        kernel_height, kernel_width = 2, 2
        nChannels = 1
        nFilters = 1
        input_shape = (image_height,image_width,nChannels)
        strides = (1,1)
        init_kernel = generate_keras_kernel(kernel_height,kernel_width,nFilters,nChannels)
        init_bias = -52.6*np.ones((nFilters,))
        kernel_initializer = initializers.constant(np.flip(init_kernel)) # [METHOD 2] keras doesn't flip the filter during the convolution; so force the array flip manually.
        bias_initializer = initializers.constant(init_bias)
        nSpikes = 2

        model = Sequential()
        model.add(Conv2D(nFilters, (kernel_height, kernel_width), strides=strides, padding='same', activation=None, use_bias=True, input_shape=input_shape, name="one", kernel_initializer=kernel_initializer, bias_initializer=bias_initializer))
        model.add(BatchNormalization())
        model.add(Spiking_BRelu(input_shape=(image_height,image_width),sharpness=1.0,name="spike"))
        mock_image = generate_mock_image(image_height,image_width,nChannels).astype(float)
        calculated = model.layers[0](mock_image)[0,:,:,:].numpy().tolist() # gives the output of the input through this layer only
        feature_extractor = Model(inputs=model.inputs, outputs=[layer.output for layer in model.layers]) # gives cumalative output of the input through this and previous layers.
        assert False

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

# Auxillary/Helper functions
def keras_convolve2d(image,kernel,strides=(1,1),mode="same"):
    '''
        Custom function that uses scipy.signal.convove2d to reproduce the Keras 2D convolution result with strides
    '''
    return np.flip(convolve2d(np.flip(image),np.flip(kernel),mode=mode))[::strides[0],::strides[1]]

def keras_convolve2d_4dinput(image,kernel,strides=(1,1),mode="same",data_format="channels_last",filters=1):
    '''
        Custom function that uses scipy.signal.convove2d to reproduce the Keras 2D convolution result with strides. Attempts to
        handle the number of channels in image and the number of kernels per filter to calculating the result.
    '''
    if data_format.lower() == "channels_last":
        if image.ndim == 3:
            height, width, nChannels = image.shape
        elif image.ndim == 4:
            batch_size, height, width, nChannels = image.shape
    elif data_format.lower() == "channels_first":
        if image.ndim == 3:
            nChannels, height, width = image.shape
        elif image.ndim == 4:
            batch_size, nChannels, height, width = image.shape
    else:
        raise ValueError("Unknown 'data_format' passed to 'keras_convolve2d_4dinput.")

    if data_format.lower() == "channels_last":
        conv2d_answer = np.zeros((height,width,filters))
        for filter in np.arange(filters):
            for channel in np.arange(nChannels):
                id = np.ravel_multi_index((channel,filter),(nChannels,filters))
                print(f"filter: {filter:2d}  channel: {channel:2d}  linearized index: {id:2d}")
                conv2d_answer[:,:,filter] += keras_convolve2d(image[0,:,:,channel],kernel[0,:,:,id],strides,mode) # update zero index in first array position to handle batch_size
    elif data_format.lower() == "channels_first":
        conv2d_answer = np.zeros((filters,height,width))
        for filter in np.arange(filters):
            for channel in np.arange(nChannels):
                id = np.ravel_multi_index((channel,filter),(nChannels,filters))
                conv2d_answer[filter] += keras_convolve2d(image[0,channel,:,:],kernel[0,id,:,:],strides,mode) # update zero index in first array position to handle batch_size

    return conv2d_answer[::strides[0],::strides[1]]

def generate_keras_kernel(nRows,nCols,nFilters,nChannels):
    '''
        Generates an initial kernel (weights) for Keras 2D convolution layer. This essentially, creates an array from an integer sequence (1,2,3,4...) into a format 
        acceptable for the Keras model. The reordering of the columns is so that Keras applies the kernels in (my desired) sequential integer sequence. For instance,
        given two Filters with 2 kernels (channels) per filter given by

        F1 = [[[1,2],[3,4]], [[5,6],[7,8]]]
        F2 = [[[9,10],[11,12]], [[13,14],[15,16]]]

        image = [[[1,2],[3,4]], [[5,6],[7,8]]]

        To respect the filter's two kernels (channels), you must reorder the columns in the Keras kernel so that the first kernel in all filters come first then the second
        kernel in all filters, all the way up to the last kernel in all filters.

        Use the image and filters above, first filter applied to the image should give a 2D convolution result equal to [[184,112],[136,80]]. Similarly, the second filter
        applied to the image gives a 2D convolution result equal to [[472,272],[312,176]].

        Returns a 4d tensor
    '''
    column_permutations = np.concatenate((np.arange(0,nFilters*nChannels,2),np.arange(1,nFilters*nChannels,2)))
    return np.arange(1,nFilters*nChannels*nRows*nCols+1).reshape(nRows*nCols,nFilters*nChannels,order='F')[:,column_permutations].reshape(1,nRows,nCols,nFilters*nChannels)

def generate_mock_image(nRows,nCols,nChannels):
    '''
        Generates a mock image of integers in a sequence. The sequence is from 1 to nRows*nCols*nChannels.

        Returns a 4d tensor
    '''
    return np.arange(1,nRows*nCols*nChannels+1).reshape(nRows*nCols,nChannels,order='F').reshape(1,nRows,nCols,nChannels)