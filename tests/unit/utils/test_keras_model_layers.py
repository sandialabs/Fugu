# isort: skip_file
import numpy as np
import pytest

from fugu.utils.keras_helpers import keras_convolve2d, keras_convolve2d_4dinput, generate_keras_kernel, generate_mock_image
from scipy.signal import convolve2d
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras import Model, initializers

# Turn off black formatting for this file
# fmt: off

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
        image = generate_mock_image(2,2,1)
        calculated = feature_extractor(image)[0,:,:,0].numpy().tolist()
        expected = [[20.,16.],[24.,16.]]
        assert expected == calculated

    @pytest.mark.parametrize("strides,expected", [((1,1), [[20.,16.],[24.,16.]]), ((1,2), [[20.],[24.]]), ((2,1), [[20.,16.]]), ((2,2), [[20.]])])
    def test_simple_same_mode_conv2d_strides_2x2_image(self, strides, expected):
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

        image = generate_mock_image(2,2,1)
        calculated = feature_extractor(image)[0,:,:,0].numpy().tolist()
        assert expected == calculated

    def test_simple_valid_mode_conv2d_strides_3x3_image(self):
        '''
            image = [[1,2,3],[4,5,6],[7,8,9]]
            kernel = [[1,2],[3,4]]

            Answers
             - padding="same", strides=(1,1) gives [[23,33,24],[53,63,42],[52,59,36]]
             - padding="valid", strides=(1,1) gives [[23,33],[53,63]]


            Note: Must flip the input kernel matrix parameter passed to Keras model layer
            so that the Keras convolution result reproduces the scipy.signal.convolve2d result.

        '''
        input_shape = (3,3,1)
        init_kernel = np.reshape(np.arange(1,5),(1,2,2,1))
        init_bias = np.zeros((1,))
        kernel_initializer = initializers.constant(np.flip(init_kernel))
        bias_initializer = initializers.constant(init_bias)
        mock_image = generate_mock_image(3,3,1)

        model11 = Sequential()
        model11.add(Conv2D(1, (2, 2), strides=(1,1), padding='valid', activation=None, use_bias=True, input_shape=input_shape, name="one", kernel_initializer=kernel_initializer, bias_initializer=bias_initializer))
        feature_extractor11 = Model(inputs=model11.inputs, outputs=model11.get_layer(name="one").output)
        calculated = feature_extractor11(mock_image)[0,:,:,0].numpy().tolist()
        expected = [[23.,33.],[53.,63.]]
        assert expected == calculated

        model12 = Sequential()
        model12.add(Conv2D(1, (2, 2), strides=(1,2), padding='valid', activation=None, use_bias=True, input_shape=input_shape, name="one", kernel_initializer=kernel_initializer, bias_initializer=bias_initializer))
        feature_extractor12 = Model(inputs=model12.inputs, outputs=model12.get_layer(name="one").output)
        calculated = feature_extractor12(mock_image)[0,:,:,0].numpy().tolist()
        expected = [[23.],[53.]]
        assert expected == calculated

        model21 = Sequential()
        model21.add(Conv2D(1, (2, 2), strides=(2,1), padding='valid', activation=None, use_bias=True, input_shape=input_shape, name="one", kernel_initializer=kernel_initializer, bias_initializer=bias_initializer))
        feature_extractor21 = Model(inputs=model21.inputs, outputs=model21.get_layer(name="one").output)
        calculated = feature_extractor21(mock_image)[0,:,:,0].numpy().tolist()
        expected = [[23.,33.]]
        assert expected == calculated

        model22 = Sequential()
        model22.add(Conv2D(1, (2, 2), strides=(2,2), padding='valid', activation=None, use_bias=True, input_shape=input_shape, name="one", kernel_initializer=kernel_initializer, bias_initializer=bias_initializer))
        feature_extractor22 = Model(inputs=model22.inputs, outputs=model22.get_layer(name="one").output)
        calculated = feature_extractor22(mock_image)[0,:,:,0].numpy().tolist()
        expected = [[23.]]
        assert expected == calculated

    def test_simple_multichannel_conv2d(self):
        '''
            image = [[1,2],[3,4]]
            kernel = [[[1,5],[2,6]],[[3,7],[4,8]]] (2 channels)
        '''
        nChannels = 2
        input_shape = (2,2,nChannels)
        init_kernel = np.moveaxis(np.arange(1,9).reshape(1,2,2,nChannels),1,3)
        init_bias = np.zeros((1,))
        kernel_initializer = initializers.constant(np.flip(init_kernel,(0,1,2)))
        bias_initializer = initializers.constant(init_bias)
        mode = "same"
        strides = (1,1)
        mock_image = generate_mock_image(2,2,nChannels)

        model = Sequential()
        model.add(Conv2D(1, (2, 2), strides=strides, padding=mode, activation=None, use_bias=True, input_shape=input_shape, name="one", kernel_initializer=kernel_initializer, bias_initializer=bias_initializer))

        # feature_extractor = Model(inputs=model.inputs, outputs=model.get_layer(name="one").output)
        feature_extractor = Model(inputs=model.inputs, outputs=[layer.output for layer in model.layers])
        
        calculated = feature_extractor(mock_image)[0,:,:,0].numpy().tolist()
        expected = keras_convolve2d(mock_image[0,:,:,0],init_kernel[0,:,:,0],strides,mode) + keras_convolve2d(mock_image[0,:,:,1],init_kernel[0,:,:,1],strides,mode)
        assert expected.tolist() == calculated

    def test_multifilter_conv2d(self):
        '''
            image = []
            kernel = []

            3 filters with 1 kernel per filter
        '''
        nFilters = 3
        nChannels = 1
        input_shape = (2,2,nChannels)
        # init_kernel = np.repeat(np.arange(1,5),nFilters).reshape(1,2,2,nFilters)
        init_kernel = np.moveaxis(np.arange(1,13).reshape(1,nFilters,2,2),1,3)
        init_filter = init_kernel
        filters = np.arange(1,13).reshape(nFilters,2,2)
        init_bias = np.zeros((nFilters,))
        kernel_initializer = initializers.constant(np.flip(init_kernel,(0,1,2)))
        bias_initializer = initializers.constant(init_bias)
        mode = "same"
        strides = (1,1)

        model = Sequential()
        model.add(Conv2D(nFilters, (2, 2), strides=strides, padding=mode, activation=None, use_bias=True, input_shape=input_shape, name="one", kernel_initializer=kernel_initializer, bias_initializer=bias_initializer))

        # feature_extractor = Model(inputs=model.inputs, outputs=model.get_layer(name="one").output)
        feature_extractor = Model(inputs=model.inputs, outputs=[layer.output for layer in model.layers])
        
        mock_image = generate_mock_image(2,2,nChannels)
        calculated = feature_extractor(mock_image)[0,:,:,:].numpy().tolist()
        # expected = [[[20.,60.,100.],[16.,40.,64.]],[[24.,52.,80.],[16.,32.,48.]]]
        expected = keras_convolve2d_4dinput(mock_image,init_kernel,strides=strides,mode=mode,filters=nFilters).tolist()
        assert expected == calculated

    def test_explicit_multifilter_multichannel_conv2d(self):
        '''
            The initial kernel must be structured so that all first channel (kernel) values in each filter come first, followed by all second channel (kernel) values in each filter, then third channel (kernel) values in each filter. 
            This process is repeated until all channels (kernels) in each filter have been covered. Here, a filter is composed of kernels. The number of kernels in a filter must match the number of channels in the input image. Hence,
            for this reason, we can use number of kernels per filter interchangeably with the number of channels per filter. At the end of the day, the number of filters or number of channels is simply the depth of filter (or image).

        '''
        nRows, nCols = 2, 2
        nFilters = 3
        nChannels = 2
        input_shape = (nRows,nCols,nChannels)
        init_kernel = generate_keras_kernel(nRows,nCols,nFilters,nChannels)
        init_bias = np.zeros((nFilters,))
        kernel_initializer = initializers.constant(np.flip(init_kernel,(0,1,2)))
        bias_initializer = initializers.constant(init_bias)
        mode = "same"
        strides = (1,1)

        model = Sequential()
        model.add(Conv2D(nFilters, (2, 2), strides=strides, padding=mode, activation=None, use_bias=True, input_shape=input_shape, name="one", kernel_initializer=kernel_initializer, bias_initializer=bias_initializer))

        # feature_extractor = Model(inputs=model.inputs, outputs=model.get_layer(name="one").output)
        feature_extractor = Model(inputs=model.inputs, outputs=[layer.output for layer in model.layers])
        
        mock_image = generate_mock_image(nRows,nCols,nChannels)
        calculated = feature_extractor(mock_image)[0,:,:,:].numpy().tolist()

        # expected result
        expected = keras_convolve2d_4dinput(mock_image,init_kernel,strides=strides,mode=mode,filters=nFilters).tolist()
        assert expected == calculated

    @pytest.mark.parametrize("strides",[(1,1),(1,2),(2,1),(2,2)])
    @pytest.mark.parametrize("nFilters", [3,4])
    @pytest.mark.parametrize("nChannels",[2,3])
    def test_multifilter_multichannel_conv2d(self,strides,nFilters,nChannels):
        '''
            The initial kernel must be structured so that all first channel (kernel) values in each filter come first, followed by all second channel (kernel) values in each filter, then third channel (kernel) values in each filter. 
            This process is repeated until all channels (kernels) in each filter have been covered. Here, a filter is composed of kernels. The number of kernels in a filter must match the number of channels in the input image. Hence,
            for this reason, we can use number of kernels per filter interchangeably with the number of channels per filter. At the end of the day, the number of filters or number of channels is simply the depth of filter (or image).

        '''
        nRows, nCols = 2, 2
        input_shape = (nRows,nCols,nChannels)
        init_kernel = generate_keras_kernel(nRows,nCols,nFilters,nChannels)
        init_bias = np.zeros((nFilters,))
        kernel_initializer = initializers.constant(np.flip(init_kernel,(0,1,2)))
        bias_initializer = initializers.constant(init_bias)
        mode = "same"

        model = Sequential()
        model.add(Conv2D(nFilters, (2, 2), strides=strides, padding=mode, activation=None, use_bias=True, input_shape=input_shape, name="one", kernel_initializer=kernel_initializer, bias_initializer=bias_initializer))

        # feature_extractor = Model(inputs=model.inputs, outputs=model.get_layer(name="one").output)
        feature_extractor = Model(inputs=model.inputs, outputs=[layer.output for layer in model.layers])
        
        mock_image = generate_mock_image(nRows,nCols,nChannels)
        calculated = feature_extractor(mock_image)[0,:,:,:].numpy().tolist()

        # expected result
        expected = keras_convolve2d_4dinput(mock_image,init_kernel,strides=strides,mode=mode,filters=nFilters).tolist()
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
    @pytest.mark.parametrize("mock_conv2d_output,pool_size,pool_strides,expected", [(np.arange(1,10).reshape((1,3,3,1)), (1,1), (1,1), [[30.,16.],[24.,16.]]),
                                                                                    (np.arange(1,10).reshape((1,3,3,1)), (2,2), (1,1), [[24]])])
    @pytest.mark.xfail(reason="Not implemented.")
    def test_simple_multilayer_model(self, mock_conv2d_output, pool_size, pool_strides, expected):
        input_shape = (2,2,1)
        init_kernel = np.reshape(np.arange(1,5),(1,2,2,1))
        init_bias = np.zeros((1,))
        kernel_initializer = initializers.constant(np.flip(init_kernel))
        bias_initializer = initializers.constant(init_bias)

        model = Sequential()
        model.add(Conv2D(1, (2, 2), strides=(1,1), padding='same', activation=None, use_bias=True, input_shape=input_shape, name="one", kernel_initializer=kernel_initializer, bias_initializer=bias_initializer))
        model.add(MaxPooling2D(pool_size=pool_size, strides=pool_strides, padding='same', name="two"))
        # model.add(MaxPooling2D(pool_size=(2,2), strides=(1,1), padding='same', name="two"))

        # feature_extractor = Model(inputs=model.inputs, outputs=model.get_layer(name="two").output)
        feature_extractor = Model(inputs=model.inputs, outputs=[layer.output for layer in model.layers])

        mock_image = np.arange(1,5).reshape(1,2,2,1)
        calculated = feature_extractor(mock_image)[1][0,:,:,0].numpy().tolist()
        assert expected == calculated
