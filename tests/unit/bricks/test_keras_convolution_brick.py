# isort: skip_file
import numpy as np
import pytest
import logging
from contextlib import nullcontext as does_not_raise

from fugu.backends import snn_Backend
from fugu.bricks.keras_convolution_bricks import keras_convolution_2d_4dinput
from fugu.bricks.input_bricks import BaseP_Input
from fugu.scaffold import Scaffold
from fugu.utils.keras_helpers import keras_convolve2d, keras_convolve2d_4dinput, generate_keras_kernel, generate_mock_image, keras_convolution2d_output_shape_4dinput
from ..helpers import ConvolutionParams, tensorflow_keras_conv2d_answer

from scipy.signal import convolve2d

# Turn off black formatting for this file
# fmt: off

def get_spiked_thresholds(image,kernel,strides,mode,nSpikes):
    # image = self.pvector
    # kernel = self.filters
    strided_answer = keras_convolve2d(image,kernel,strides,mode)

    subt = np.zeros(np.size(strided_answer))
    subt[:nSpikes] = 0.1
    subt = np.reshape(subt, strided_answer.shape)
    spiked_thresholds = strided_answer - subt

    return spiked_thresholds

def get_biased_thresholds(image,kernel,strides,mode,biases):
    # image = self.pvector
    # kernel = self.filters
    strided_answer = keras_convolve2d(image,kernel,strides,mode)

    biased_thresholds = strided_answer + biases if biases is not None else strided_answer
    return biased_thresholds

def get_output_shape(image,kernel,strides,mode):
    # image = self.pvector
    # kernel = self.filters
    strided_answer = keras_convolve2d(image,kernel,strides,mode)
    return strided_answer.shape

@pytest.mark.keras
@pytest.mark.keras_convolution
class Test_KerasConvolution2D_4dinput:
    def setup_method(self):
        image_height, image_width = 3, 3
        kernel_height, kernel_width = 2, 2
        nChannels = 2
        nFilters = 3

        self.basep = 4
        self.bits = 4
        self.pvector = generate_mock_image(image_height,image_width,nChannels)
        self.filters = generate_keras_kernel(kernel_height,kernel_width,nFilters,nChannels)
        self.pshape = np.array(self.pvector).shape
        self.filters_shape = np.array(self.filters).shape
        self.mode = "same"
        self.strides = (1,1)
        self.biases = np.zeros((nFilters,))
        self.nChannels = nChannels
        self.nFilters = nFilters

    @pytest.mark.parametrize("bias",[-63,-62,-58])
    def test_explicit_simple_convolution_with_bias(self,bias):
        '''
            Convolution answer is
                  [[23., 33., 24.],
                   [53., 63., 42.],
                   [52., 59., 36.]]
        '''
        image_height, image_width = 3, 3
        kernel_height, kernel_width = 2, 2
        nChannels = 1
        nFilters = 1

        convo_obj = ConvolutionParams(image_height=image_height,image_width=image_width,nChannels=nChannels,
                                      kernel_height=kernel_height,kernel_width=kernel_width,nFilters=nFilters)
        self.pvector = generate_mock_image(image_height,image_width,nChannels)
        self.filters = generate_keras_kernel(kernel_height,kernel_width,nFilters,nChannels)
        self.pshape = np.array(self.pvector).shape
        self.filters_shape = np.array(self.filters).shape
        self.nChannels = nChannels
        self.nFilters = nFilters
        self.biases = np.array([bias]) # leads to 2 spike at g22 and g32

        output_shape = keras_convolution2d_output_shape_4dinput(self.pvector,self.filters,self.strides,self.mode,self.nFilters)
        thresholds = 0.5*np.ones(output_shape).reshape(1,*output_shape)
        keras_convolution_answer = keras_convolve2d_4dinput(self.pvector,self.filters,strides=self.strides,mode=self.mode,filters=self.nFilters)
        expected_spike_count = (keras_convolution_answer + self.biases > thresholds).sum().astype(int)

        result = self.run_convolution_2d(thresholds)
        calculated_spike_count = len(result[result['time'] > 0].index)

        assert expected_spike_count == calculated_spike_count

    @pytest.mark.parametrize("nSpikes,biases",[(3,[-471, -1207, -1943]),(4,[-471, -1107, -1943]),(0,[-472, -1208, -1944])])
    def test_explicit_same_convolution_with_bias(self,nSpikes,biases):
        '''
            Given image and kernel provided in setup_method, the convolution ("same") answer is
                        Filter 1 Out            Filter 2 Out            Filter 3 Out
                  [[ 328.,  364.,  210.], [[ 808.,  908.,  498.], [[1288., 1452.,  786.], 
                   [ 436.,  472.,  270.],  [1108., 1208.,  654.],  [1780., 1944., 1038.], 
                   [ 299.,  321.,  180.]], [ 683.,  737.,  396.]], [1067., 1153.,  612.]])
        '''
        self.biases = np.array(biases)
        output_shape = keras_convolution2d_output_shape_4dinput(self.pvector,self.filters,self.strides,self.mode,self.nFilters)
        thresholds = 0.5*np.ones(output_shape).reshape(1,*output_shape)
        keras_convolution_answer = keras_convolve2d_4dinput(self.pvector,self.filters,strides=self.strides,mode=self.mode,filters=self.nFilters)
        expected_spike_count = (keras_convolution_answer + self.biases > thresholds).sum().astype(int)

        result = self.run_convolution_2d(thresholds)
        calculated_spike_count = len(result[result['time'] > 0].index)

        assert expected_spike_count == calculated_spike_count

    @pytest.mark.parametrize("nSpikes,biases,strides",[(7,[-320, -682, -1152],(1,2)),(12,[-298, -682, -1037],(2,1)),(0,[-472, -1208, -1944],(1,2)),(9,[-322, -736, -1066],(2,1)),(0,[-328, -1208, -1944],(2,2))])
    def test_explicit_same_convolution_with_bias_and_strides(self,nSpikes,biases,strides):
        '''
            Given image and kernel provided in setup_method, the convolution ("same") answer is
                        Filter 1 Out            Filter 2 Out            Filter 3 Out
                  [[ 328.,  364.,  210.], [[ 808.,  908.,  498.], [[1288., 1452.,  786.],
                   [ 436.,  472.,  270.],  [1108., 1208.,  654.],  [1780., 1944., 1038.],
                   [ 299.,  321.,  180.]], [ 683.,  737.,  396.]], [1067., 1153.,  612.]])
        '''
        self.strides = strides
        self.biases = np.array(biases)
        output_shape = keras_convolution2d_output_shape_4dinput(self.pvector,self.filters,self.strides,self.mode,self.nFilters)
        thresholds = 0.5*np.ones(output_shape).reshape(1,*output_shape)
        keras_convolution_answer = keras_convolve2d_4dinput(self.pvector,self.filters,strides=self.strides,mode=self.mode,filters=self.nFilters)
        expected_spike_count = (keras_convolution_answer + self.biases > thresholds).sum().astype(int)

        result = self.run_convolution_2d(thresholds)
        calculated_spike_count = len(result[result['time'] > 0].index)

        assert expected_spike_count == calculated_spike_count

    @pytest.mark.parametrize("nSpikes,biases",[(3,[-471, -1207, -1943]),(4,[-471, -1107, -1943]),(0,[-472, -1208, -1944]),(2,[-435, -1208, -1944]),(3,[-472, -907, -1944]),(2,[-472, -1208, -1779])])
    def test_explicit_valid_convolution_with_bias(self,nSpikes,biases):
        '''
            Given image and kernel provided in setup_method, the convolution ("valid") answer is
                        Filter 1 Out            Filter 2 Out            Filter 3 Out
                  [[ 328.,  364.],        [[ 808.,  908.],        [[1288., 1452.],
                   [ 436.,  472.]],        [1108., 1208.]],        [1780., 1944.]])
        '''
        self.mode = "valid"
        self.biases = np.array(biases)
        output_shape = keras_convolution2d_output_shape_4dinput(self.pvector,self.filters,self.strides,self.mode,self.nFilters)
        thresholds = 0.5*np.ones(output_shape).reshape(1,*output_shape)
        keras_convolution_answer = keras_convolve2d_4dinput(self.pvector,self.filters,strides=self.strides,mode=self.mode,filters=self.nFilters)
        expected_spike_count = (keras_convolution_answer + self.biases > thresholds).sum().astype(int)

        result = self.run_convolution_2d(thresholds)
        calculated_spike_count = len(result[result['time'] > 0].index)

        assert expected_spike_count == calculated_spike_count

    @pytest.mark.parametrize("nSpikes,biases,strides",[(3,[-363, -907, -1451],(1,2)),(6,[-327, -807, -1287],(2,1)),(2,[-327, -808, -1287],(2,2))])
    def test_explicit_valid_convolution_with_bias_and_strides(self,nSpikes,biases,strides):
        '''
            Given image and kernel provided in setup_method, the convolution ("valid") answer is
                        Filter 1 Out            Filter 2 Out            Filter 3 Out
                  [[ 328.,  364.],        [[ 808.,  908.],        [[1288., 1452.],
                   [ 436.,  472.]],        [1108., 1208.]],        [1780., 1944.]])
        '''
        self.mode = "valid"
        self.strides = strides
        self.biases = np.array(biases)
        output_shape = keras_convolution2d_output_shape_4dinput(self.pvector,self.filters,self.strides,self.mode,self.nFilters)
        thresholds = 0.5*np.ones(output_shape).reshape(1,*output_shape)
        keras_convolution_answer = keras_convolve2d_4dinput(self.pvector,self.filters,strides=self.strides,mode=self.mode,filters=self.nFilters)
        expected_spike_count = (keras_convolution_answer + self.biases > thresholds).sum().astype(int)

        result = self.run_convolution_2d(thresholds)
        calculated_spike_count = len(result[result['time'] > 0].index)

        assert expected_spike_count == calculated_spike_count

    @pytest.mark.skip(reason="Slow test. Doesn't show 'dots' as one dot tests all matrix entries per stride value. Additionally, this is a duplicate of exhaustive test below, which does show 'dots' for each matrix entry per mode, but demonstrates the explicit version of that (exhaustive) test.")
    @pytest.mark.parametrize("strides", [(1,1),(1,2),(2,1),(2,2),(1,3),(3,1),(2,3),(3,2),(3,3)])
    def test_5x5_image_same_mode_with_strides_and_biases(self,strides):
        '''
            Filter 1 Out
          [[ 772.,  808.,  844.,  880.,  490.],
           [ 952.,  988., 1024., 1060.,  590.],
           [1132., 1168., 1204., 1240.,  690.],
           [1312., 1348., 1384., 1420.,  790.],
           [ 847.,  869.,  891.,  913.,  500.]]

            Filter 2 Out
          [[1828., 1928., 2028., 2128., 1130.],
           [2328., 2428., 2528., 2628., 1390.],
           [2828., 2928., 3028., 3128., 1650.],
           [3328., 3428., 3528., 3628., 1910.],
           [1935., 1989., 2043., 2097., 1100.]]

            Filter 3 Out
          [[2884., 3048., 3212., 3376., 1770.],
           [3704., 3868., 4032., 4196., 2190.],
           [4524., 4688., 4852., 5016., 2610.],
           [5344., 5508., 5672., 5836., 3030.],
           [3023., 3109., 3195., 3281., 1700.]]
        '''
        image_height, image_width = 5, 5
        kernel_height, kernel_width = 2, 2
        nChannels = 2
        nFilters = 3

        self.basep = 4
        self.bits = 4
        self.pvector = generate_mock_image(image_height,image_width,nChannels)
        self.filters = generate_keras_kernel(kernel_height,kernel_width,nFilters,nChannels)
        self.pshape = np.array(self.pvector).shape
        self.filters_shape = np.array(self.filters).shape
        self.mode = "same"
        self.strides = strides
        self.biases = np.zeros((nFilters,))
        self.nChannels = nChannels
        self.nFilters = nFilters

        keras_convolution_answer = keras_convolve2d_4dinput(self.pvector,self.filters,strides=self.strides,mode=self.mode,filters=self.nFilters)
        thresholds = 0.5*np.ones(keras_convolution_answer.shape).reshape(1,*keras_convolution_answer.shape)

        biases_list = np.sort(keras_convolution_answer,axis=None)
        biases_list = np.flip(np.append(biases_list, biases_list)).astype(float)
        biases_list[1:] -= 0.6
        for k, bias in enumerate(biases_list):
            self.biases = -bias * np.ones((nFilters,))
            result = self.run_convolution_2d(thresholds)

            expected_spike_count = (keras_convolution_answer + self.biases > thresholds).sum().astype(int)
            calculated_spike_count = len(result[result['time'] > 0].index)

            assert expected_spike_count == calculated_spike_count

    def generate_all_values_for_exhaustive_test(self):
        '''
            Helper function to generate all the possible parameters for a 5x5 image with 2 channels and a 3 filters with a 2x2 kernel at different strides and modes{"same","valid"}.
        '''
        from ..helpers import ArraySequence, IntegerSequence
        from tensorflow import constant as tf_constant
        from tensorflow.keras import Model, initializers
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Conv2D

        runSubsetOn = True # Run only a subset of the full exhaustive parameter space; otherwise runs every parameter sequence
        subset_size = 100  # Note: Full parameter space size is 510
        class Empty:
            pass

        image_height, image_width = 5, 5
        kernel_height, kernel_width = 2, 2
        nChannels = 2
        nFilters = 3

        aself = Empty()
        aself.basep = 4
        aself.bits = 4
        aself.pvector = generate_mock_image(image_height,image_width,nChannels)
        aself.filters = generate_keras_kernel(kernel_height,kernel_width,nFilters,nChannels)
        aself.pshape = np.array(aself.pvector).shape
        aself.filters_shape = np.array(aself.filters).shape
        aself.biases = np.zeros((nFilters,))
        aself.nChannels = nChannels
        aself.nFilters = nFilters
        thresholds = 0.5

        arguments = []
        for mode in ["same", "valid"]:
            aself.mode = mode
            for strides in [(1,1),(1,2),(2,1),(2,2),(1,3),(3,1),(2,3),(3,2),(3,3)]:
                aself.strides = strides

                model = Sequential()
                model.add(Conv2D(aself.nFilters, aself.filters.shape[:2], strides=aself.strides, padding=aself.mode, activation=None, use_bias=True, 
                                input_shape=aself.pshape[1:], name="conv2d", kernel_initializer=ArraySequence(np.flip(aself.filters,(0,1))), bias_initializer=ArraySequence(aself.biases)))
                feature_extractor = Model(inputs=model.inputs, outputs=[layer.output for layer in model.layers])
                feature_extractor_answer = feature_extractor(aself.pvector)[0].numpy()
                keras_spike_count = (feature_extractor_answer > thresholds).astype(int).sum()

                biases_list = np.sort(feature_extractor_answer,axis=None)
                biases_list = np.flip(np.append(biases_list, biases_list[-1])).astype(float)
                biases_list[1:] -= 0.6
                for k, bias in enumerate(biases_list):
                    biases = -bias * np.ones((nFilters,))
                    nSpikes = (feature_extractor_answer + biases> thresholds).sum()
                    if runSubsetOn:
                        arguments.append((nSpikes, thresholds, biases, strides, mode, aself))
                    else:
                        yield (nSpikes, thresholds, biases, strides, mode, aself)


        if runSubsetOn:
            # randomly chose a subset of the full parameter space for the unit tests
            yield_list = np.array(arguments,dtype=[('nSpikes',int),('thresholds',object),('biases',object),('strides',tuple),('mode','<U10'),('aself',object)])
            yield_subset = np.random.choice(yield_list,subset_size,replace=False)
            for nSpikes, thresholds, biases, strides, mode, aself in yield_subset:
                yield nSpikes, thresholds, biases, strides, mode, aself


    @pytest.fixture
    def get_self(self):
        return self

    # @pytest.mark.slow
    @pytest.mark.parametrize("nSpikes, thresholds, biases, strides, mode, aself", generate_all_values_for_exhaustive_test(get_self))
    def test_exhaustive_5x5_image_all_modes_strides_biases(self,nSpikes,thresholds,biases,strides,mode,aself):
        '''
            Exhaustive test that check each matrix element in the convolution answer for correctness. This is accomplished by setting the biases to the value of the exact answer, thereby, 
            shifting the neuron voltage to 0. An output neuron spike when the neuron voltage is > 0.5.
        '''
        for property, value in vars(aself).items():
            if property[0] == '_':
                continue
            setattr(self,property,value)

        self.biases = biases
        self.strides = strides
        self.mode = mode
        result = self.run_convolution_2d(thresholds)
        calculated_spike_count = len(result[result['time'] > 0].index)
        expected_spike_count = nSpikes
        assert expected_spike_count == calculated_spike_count

    @pytest.mark.parametrize("image_size", [(3,3),(5,5)])
    @pytest.mark.parametrize("kernel_size", [(2,2),(3,3)])
    @pytest.mark.parametrize("strides", [(1,1),(1,2),(2,1),(2,2),(1,3),(3,1),(2,3),(3,2),(3,3)])
    @pytest.mark.parametrize("mode", ["same", "valid"])
    @pytest.mark.parametrize("nChannels,nFilters", [(2,3), (1,1)])
    def test_exhaustive_keras_convolution_brick_all_modes_strides_with_random_biases(self, image_size, kernel_size, strides, mode, nChannels, nFilters):
        self.basep = 4
        self.bits = 3
        convo_obj = ConvolutionParams(image_height=image_size[0], image_width=image_size[1], nChannels=nChannels, 
                                      kernel_height=kernel_size[0], kernel_width=kernel_size[1], nFilters=nFilters, 
                                      strides=strides, mode=mode, data_format="channels_last", biases=np.zeros((nFilters,)))
        convo_obj.biases = convo_obj.get_random_biases_within_answer_range()

        scaffold = Scaffold()
        scaffold.add_brick(BaseP_Input(convo_obj.mock_image,p=self.basep,bits=self.bits,collapse_binary=False,name="I",time_dimension=False),"input")
        scaffold.add_brick(keras_convolution_2d_4dinput(convo_obj.input_shape,convo_obj.filters,convo_obj.thresholds,self.basep,self.bits,name="convolution_",mode=convo_obj.mode,strides=convo_obj.strides,biases=convo_obj.biases),[(0, 0)],output=True)
        self.graph = scaffold.lay_bricks()
        scaffold.summary(verbose=1)
        backend = snn_Backend()
        backend_args = {}
        backend.compile(scaffold, backend_args)
        result = backend.run(5)
        calculated_spike_count = len(result[result['time'] > 0].index)

        keras_obj = tensorflow_keras_conv2d_answer(convo_obj)
        expected_spike_count = keras_obj.spike_count
        assert expected_spike_count == calculated_spike_count

    @pytest.mark.parametrize("strides", [(1,1),(1,2),(2,1),(2,2),(1,3),(3,1),(2,3),(3,2),(3,3)])
    @pytest.mark.parametrize("mode", ["same", "valid"])
    @pytest.mark.parametrize("nChannels,nFilters", [(2,3), (1,1)])
    def test_exhaustive_5x5_image_all_modes_strides_random_biases(self, strides, mode, nChannels, nFilters):
        from ..helpers import ArraySequence, IntegerSequence
        from tensorflow import constant as tf_constant
        from tensorflow.keras import Model, initializers
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Conv2D

        self.basep = 4
        self.bits = 3

        convo_obj = ConvolutionParams(image_height=5, image_width=5, nChannels=nChannels, kernel_height=2, kernel_width=2, nFilters=nFilters, strides=strides, mode=mode, biases=np.zeros((nFilters,)))
        convo_obj.biases = convo_obj.get_random_biases_within_answer_range()
        convo_obj._set_convolution_answer_boolean()
        expected_spike_count = convo_obj.answer_bool.astype(int).sum()

        keras_convolution_answer = keras_convolve2d_4dinput(convo_obj.mock_image,convo_obj.filters,strides=convo_obj.strides,mode=convo_obj.mode,filters=convo_obj.nFilters)
        keras_spike_count = (keras_convolution_answer + convo_obj.biases > convo_obj.thresholds).sum().astype(int)

        scaffold = Scaffold()
        scaffold.add_brick(BaseP_Input(convo_obj.mock_image,p=self.basep,bits=self.bits,collapse_binary=False,name="I",time_dimension=False),"input")
        scaffold.add_brick(keras_convolution_2d_4dinput(convo_obj.input_shape,convo_obj.filters,convo_obj.thresholds,self.basep,self.bits,name="convolution_",mode=convo_obj.mode,strides=convo_obj.strides,biases=convo_obj.biases),[(0, 0)],output=True)
        self.graph = scaffold.lay_bricks()
        scaffold.summary(verbose=1)
        backend = snn_Backend()
        backend_args = {}
        backend.compile(scaffold, backend_args)
        result = backend.run(5)
        calculated_spike_count = len(result[result['time'] > 0].index)
        print(convo_obj.biases)


        model = Sequential()
        model.add(Conv2D(convo_obj.nFilters, (convo_obj.kernel_height, convo_obj.kernel_width), strides=convo_obj.strides, padding=convo_obj.mode, activation=None, use_bias=True, 
                         input_shape=convo_obj.input_shape[1:], name="conv2d", kernel_initializer=ArraySequence(np.flip(convo_obj.filters,(0,1))), bias_initializer=ArraySequence(convo_obj.biases)))
        feature_extractor = Model(inputs=model.inputs, outputs=[layer.output for layer in model.layers])
        feature_extractor_answer = feature_extractor(convo_obj.mock_image)[0].numpy()
        keras_spike_count = (feature_extractor_answer > convo_obj.thresholds).astype(int).sum()

        assert keras_spike_count == calculated_spike_count

    def test_positive_biases(self):
        image_height, image_width, nChannels = 3, 3, 2
        kernel_height, kernel_width, nFilters = 2, 2, 4

        self.pvector = generate_mock_image(image_height,image_width,nChannels)
        self.filters = generate_keras_kernel(kernel_height,kernel_width,nFilters,nChannels)
        self.pshape = np.array(self.pvector).shape
        self.filters_shape = np.array(self.filters).shape
        self.nChannels = nChannels
        self.nFilters = nFilters
        self.biases = np.array([-471., -1207., -1943., 500.])

        output_shape = keras_convolution2d_output_shape_4dinput(self.pvector,self.filters,self.strides,self.mode,self.nFilters)
        thresholds = 0.5*np.ones(output_shape).reshape(1,*output_shape)
        keras_convolution_answer = keras_convolve2d_4dinput(self.pvector,self.filters,strides=self.strides,mode=self.mode,filters=self.nFilters)
        expected_spike_count = (keras_convolution_answer + self.biases > thresholds).sum().astype(int)

        result = self.run_convolution_2d(thresholds)
        calculated_spike_count = len(result[result['time'] > 0].index)

        assert expected_spike_count == calculated_spike_count

    @pytest.mark.parametrize("strides",[(1,1),(1,2),(2,1),(2,2)])
    def test_28x28_image_same_mode_with_strides_and_biases(self, strides):
        '''
            Filter 1 Out
          [[ 772.,  808.,  844.,  880.,  490.],
           [ 952.,  988., 1024., 1060.,  590.],
           [1132., 1168., 1204., 1240.,  690.],
           [1312., 1348., 1384., 1420.,  790.],
           [ 847.,  869.,  891.,  913.,  500.]]

            Filter 2 Out
          [[1828., 1928., 2028., 2128., 1130.],
           [2328., 2428., 2528., 2628., 1390.],
           [2828., 2928., 3028., 3128., 1650.],
           [3328., 3428., 3528., 3628., 1910.],
           [1935., 1989., 2043., 2097., 1100.]]

            Filter 3 Out
          [[2884., 3048., 3212., 3376., 1770.],
           [3704., 3868., 4032., 4196., 2190.],
           [4524., 4688., 4852., 5016., 2610.],
           [5344., 5508., 5672., 5836., 3030.],
           [3023., 3109., 3195., 3281., 1700.]]
        '''
        from ..helpers import ArraySequence, IntegerSequence
        from tensorflow import constant as tf_constant
        from tensorflow.keras import Model, initializers
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Conv2D

        image_height, image_width = 28, 28
        kernel_height, kernel_width = 7, 7
        nChannels = 1
        nFilters = 1

        self.basep = 3
        self.bits = 8
        self.pvector = generate_mock_image(image_height,image_width,nChannels)
        self.filters = generate_keras_kernel(kernel_height,kernel_width,nFilters,nChannels)
        self.pshape = np.array(self.pvector).shape
        self.filters_shape = np.array(self.filters).shape
        self.mode = "same"
        self.strides = strides
        self.biases = np.zeros((nFilters,))
        self.nChannels = nChannels
        self.nFilters = nFilters

        keras_convolution_answer = keras_convolve2d_4dinput(self.pvector,self.filters,strides=self.strides,mode=self.mode,filters=self.nFilters)
        thresholds = 0.5*np.ones(keras_convolution_answer.shape).reshape(1,*keras_convolution_answer.shape)
        # thresholds = np.reshape(keras_convolution_answer,(1,*keras_convolution_answer.shape)).copy() - 1.

        # self.biases = -2909.0 * np.ones((nFilters,))
        # self.biases = -50000.0 * np.ones((nFilters,))
        result = self.run_convolution_2d(thresholds, verbose_scaffold=1)

        model = Sequential()
        model.add(Conv2D(nFilters, (kernel_height, kernel_width), strides=self.strides, padding=self.mode, activation=None, use_bias=True, 
                         input_shape=self.pvector.shape[1:], name="conv2d", kernel_initializer=ArraySequence(np.flip(self.filters,(0,1))), bias_initializer=ArraySequence(self.biases)))
        feature_extractor = Model(inputs=model.inputs, outputs=[layer.output for layer in model.layers])
        feature_extractor_answer = feature_extractor(self.pvector)[0].numpy()
        keras_spike_count = (feature_extractor_answer > thresholds).astype(int).sum()

        expected_spike_count = (keras_convolution_answer + self.biases > thresholds).sum().astype(int)
        calculated_spike_count = len(self.get_convolution_neurons_result_only(result).index)

        assert expected_spike_count == calculated_spike_count

    @pytest.mark.xfail(reason="Not implemented.")
    def test_strides_parameter_handling(self):
        assert False

    @pytest.mark.xfail(reason="Not implemented.")
    def test_data_format_channels_first(self):
        assert False

    @pytest.mark.xfail(reason="Not implemented.")
    def test_data_format_channels_last(self):
        assert False

    @pytest.mark.xfail(reason="Not implemented.")
    def test_output_neuron_creation(self):
        assert False

    @pytest.mark.xfail(reason="Not implemented.")
    def test_get_output_neurons(self):
        assert False

    @pytest.mark.xfail(reason="Not implemented.")
    def test_get_output_bounds(self):
        assert False

    @pytest.mark.xfail(reason="Not implemented.")
    def test_input_output_neurons_connections(self):
        assert False

    @pytest.mark.xfail(reason="Not implemented.")
    def test_padding_size(self):
        assert False

    # Auxillary/Helper Function below
    def get_neuron_numbers(self, name_prefix):
        neuron_numbers = []
        for key in self.graph.nodes.keys():
            if key.startswith(name_prefix):
                neuron_numbers.append(self.graph.nodes[key]['neuron_number'])

        return np.array(neuron_numbers)

    def get_convolution_neurons_result_only(self, result):
        convolution_neuron_numbers = self.get_neuron_numbers('convolution_g')
        sub_result = result[result['neuron_number'].isin(convolution_neuron_numbers)]
        return sub_result

    def get_num_output_neurons(self, thresholds):
        Am, An = self.pshape[1:3]
        Bm, Bn = self.filters_shape[:2]
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

    def run_convolution_2d(self, thresholds, verbose_scaffold=1):
        scaffold = Scaffold()
        scaffold.add_brick(
            BaseP_Input(
                np.array([self.pvector]),
                p=self.basep,
                bits=self.bits,
                collapse_binary=False,
                name="I",
                time_dimension=False,
            ),
            "input",
        )
        scaffold.add_brick(
            keras_convolution_2d_4dinput(
                self.pshape,
                self.filters,
                thresholds,
                self.basep,
                self.bits,
                name="convolution_",
                mode=self.mode,
                strides=self.strides,
                biases=self.biases,
            ),
            [(0, 0)],
            output=True,
        )
        self.graph = scaffold.lay_bricks()
        scaffold.summary(verbose=verbose_scaffold)
        backend = snn_Backend()
        backend_args = {}
        backend.compile(scaffold, backend_args)
        result = backend.run(5)
        return result