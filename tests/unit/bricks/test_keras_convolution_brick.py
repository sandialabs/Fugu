# isort: skip_file
import numpy as np
import pytest
import logging
from contextlib import nullcontext as does_not_raise

from fugu.backends import snn_Backend
from fugu.bricks.keras_convolution_bricks import keras_convolution_2d, keras_convolution_2d_4dinput
from fugu.bricks.input_bricks import BaseP_Input
from fugu.scaffold import Scaffold
from fugu.utils.keras_helpers import keras_convolve2d, keras_convolve2d_4dinput, generate_keras_kernel, generate_mock_image, keras_convolution2d_output_shape_4dinput

from scipy.signal import convolve2d

# Turn off black formatting for this file
# fmt: off

class Test_KerasConvolution2D:
    def setup_method(self):
        self.basep = 2
        self.bits = 2
        self.pvector = [[1, 1], [1, 1]]
        self.filters = [[1, 2], [3, 4]]
        self.pshape = np.array(self.pvector).shape
        self.filters_shape = np.array(self.filters).shape
        self.mode = "same"
        self.strides = (1,1)
        self.biases = None

    @pytest.mark.parametrize("basep", [2])
    @pytest.mark.parametrize("bits", [2])
    @pytest.mark.parametrize("thresholds", np.arange(0.9, 11, 1))
    def test_scalar_threshold(self, basep, bits, thresholds):
        ans_thresholds = np.array(
            [[10, 6], [7, 4,]]
        )  # 2d convolution answer is [[1,3,2],[4,10,6],[3,7,4]]. Spikes fire when less than threshold. Thus subtract 0.1 so that spikes fire
        nSpikes = len(ans_thresholds[ans_thresholds > thresholds])

        self.basep = basep
        self.bits = bits
        result = self.run_convolution_2d(thresholds)

        # get output positions in result
        output_positions = self.output_spike_positions(
            self.basep, self.bits, self.pvector, self.filters, thresholds
        )
        output_mask = self.output_mask(output_positions, result)

        # Check calculations
        if nSpikes == 0:
            expected_spikes = list(np.array([]))
        else:
            expected_spikes = list(np.ones((nSpikes,)))

        calculated_spikes = list(result[output_mask].to_numpy()[:, 0])
        assert expected_spikes == calculated_spikes

    @pytest.mark.parametrize("mode", ["valid", "same"])
    @pytest.mark.parametrize("thresholds_shape", [(1,2),(2,1)])
    def test_thresholds_shape(self, mode, thresholds_shape):
        self.mode = mode
        with pytest.raises(ValueError):
            thresholds = np.ones(thresholds_shape)
            self.run_convolution_2d(thresholds)

    @pytest.mark.parametrize("mode", ["same"])
    @pytest.mark.parametrize("strides,expected", [(1,[2,2]), (2,[1,1]), ((2,1),[1,2]), ((1,2),[2,1]), ([1,2],[2,1])])
    def test_output_shape(self,mode,strides,expected):
        #TODO: Add mode="valid" to output shape check
        self.mode = mode
        self.strides = strides
        layer = keras_convolution_2d(self.pshape,self.filters,np.ones(self.filters_shape),self.basep,self.bits,name='conv',strides=self.strides)
        calculated = list(layer.output_shape)
        assert expected == calculated

    @pytest.mark.parametrize("strides,expected,expectation", 
                             [
                                 (1, (1,1), does_not_raise()), 
                                 (2, (2,2), does_not_raise()), 
                                 (2.0, (2,2), does_not_raise()), 
                                 ((2,1), (2,1), does_not_raise()), 
                                 ([1,2], (1,2), does_not_raise()), 
                                 ([2.0,2.0], (2,2), does_not_raise()), 
                                 ([1,2,3], None, pytest.raises(ValueError)),
                            ])
    def test_strides_parameter(self,strides,expected,expectation):
        with expectation:
            self.strides = strides
            layer = keras_convolution_2d(self.pshape,self.filters,np.ones(self.filters_shape),self.basep,self.bits,name='conv',strides=self.strides)
            calculated = layer.strides
            assert expected == calculated

    def test_explicit_same_mode_with_strides(self,caplog):
        '''
        mode="full" 2D convolution result is [[1,4,4],[6,20,16],[9,24,16]].

        Keras 2d convolution brick does only supports mode="same" or mode="valid". The
        "same" result should return [[20,16],[24,16]], where padding is applied only to
        the bottom and right image data; padding is NOT appended to the left and top image positions.
        '''
        caplog.set_level(logging.DEBUG)
        self.mode = "same"
        self.basep = 3 #basep
        self.bits = 3 #bits
        self.pvector = [[1, 2], [3, 4]]
        self.filters = [[1, 2], [3, 4]]
        self.pshape = np.array(self.pvector).shape
        self.filters_shape = np.array(self.filters).shape

        # manually set strides, thresholds, and expected values
        self.strides = (1,1) # answer is [[20,16],[24,16]]
        thresholds = np.array([[20,15.9],[24,15.9]])
        expected_spikes = [1, 1]
        result = self.run_convolution_2d(thresholds)
        assert expected_spikes == self.calculated_spikes(thresholds,result)

        self.strides = (2,1) # answer is [[20,16]]
        thresholds = np.array([[20,15.9]])
        expected_spikes = [1]
        result = self.run_convolution_2d(thresholds)
        assert expected_spikes == self.calculated_spikes(thresholds,result)

        self.strides = (1,2) # answer is [[20],[24]]
        thresholds = np.array([[20],[23.9]])
        expected_spikes = [1]
        result = self.run_convolution_2d(thresholds)
        assert expected_spikes == self.calculated_spikes(thresholds,result)

        self.strides = (2,2) # answer is [[20]]
        thresholds = np.array([[19.9]])
        expected_spikes = [1]
        result = self.run_convolution_2d(thresholds)
        assert expected_spikes == self.calculated_spikes(thresholds,result)

    @pytest.mark.parametrize("basep", [2, 3, 4, 5, 6, 7, 8, 9])
    @pytest.mark.parametrize("bits", [2, 3, 4, 7, 8])
    @pytest.mark.parametrize("strides,nSpikes",
                             [
                                 ((1,1), 0),
                                 ((1,1), 1),
                                 ((1,1), 2),
                                 ((1,1), 3),
                                 ((1,1), 4),
                                 ((1,2), 0),
                                 ((1,2), 1),
                                 ((1,2), 2),
                                 ((2,1), 0),
                                 ((2,1), 1),
                                 ((2,1), 2),
                                 ((2,2), 0),
                                 ((2,2), 1),
                             ])
    def test_same_mode_with_strides(self,caplog,basep,bits,strides,nSpikes):
        caplog.set_level(logging.DEBUG)
        self.basep = basep
        self.bits = bits
        self.strides = strides
        self.mode = "same"

        thresholds = get_spiked_thresholds(self.pvector,self.filters,self.strides,self.mode,nSpikes)
        result = self.run_convolution_2d(thresholds)

        assert self.expected_spikes(nSpikes) == self.calculated_spikes(thresholds,result)

    def test_3x3_image_same_mode_with_strides(self):
        self.basep = 3
        self.bits = 3
        self.pvector = [[1,2,3],[4,5,6],[7,8,9]]
        self.pshape = np.array(self.pvector).shape
        self.filters = [[1, 2], [3, 4]]
        self.filters_shape = np.array(self.filters).shape

        # manually set strides, thresholds, and expected values
        self.strides = (1,1) # answer is [[23,33,24],[53,63,42],[52,59,36]]
        thresholds = np.array([[23,33,24],[53,62.9,42],[52,58.9,36]])
        expected_spikes = [1, 1]
        result = self.run_convolution_2d(thresholds)
        assert expected_spikes == self.calculated_spikes(thresholds,result)

        self.strides = (2,2) # answer is [[23,24],[52,36]]
        thresholds = np.array([[23,23.9],[51.9,36]])
        expected_spikes = [1, 1]
        result = self.run_convolution_2d(thresholds)
        assert expected_spikes == self.calculated_spikes(thresholds,result)

    def test_2x2_image_same_mode_with_strides(self):
        self.basep = 3
        self.bits = 3
        self.pvector = [[1 ,2] ,[3, 4]]
        self.pshape = np.array(self.pvector).shape
        self.filters = [[1, 2], [3, 4]]
        self.filters_shape = np.array(self.filters).shape

        # manually set strides, thresholds, and expected values
        self.strides = (1,1) # answer is [[20,16],[24,16]]
        thresholds = np.array([[20,15.9],[24,15.9]])
        expected_spikes = [1, 1]
        result = self.run_convolution_2d(thresholds)
        assert expected_spikes == self.calculated_spikes(thresholds,result)

        self.strides = (1,1) # answer is [[20],[24]]
        thresholds = np.array([[19.9],[23.9]])
        expected_spikes = [1, 1]
        result = self.run_convolution_2d(thresholds)
        assert expected_spikes == self.calculated_spikes(thresholds,result)

        self.strides = (2,1) # answer is [[20,16]]
        thresholds = np.array([[20,15.9]])
        expected_spikes = [1]
        result = self.run_convolution_2d(thresholds)
        assert expected_spikes == self.calculated_spikes(thresholds,result)

        self.strides = (2,2) # answer is [[20]]
        thresholds = np.array([[19.9]])
        expected_spikes = [1]
        result = self.run_convolution_2d(thresholds)
        assert expected_spikes == self.calculated_spikes(thresholds,result)

    def test_explicit_valid_mode_with_strides(self):
        '''
        image=[[1,2,3],[4,5,6],[7,8,9]]
        kernel=[[1,2],[3,4]]

        mode="full" 2D convolution result is [[1,4,7,6],[7,23,33,24],[19,53,63,42],[21,52,59,36]].

        Keras 2d convolution brick does only supports mode="same" or mode="valid". The
        "valid" result should return [[23,33],[53,63]] with strides=(1,1).
        '''
        self.basep = 2
        self.bits = 4
        self.mode = "valid"
        self.pvector = np.arange(1,10).reshape(3,3)
        self.pshape = self.pvector.shape

        # manually set strides, thresholds, and expected values
        self.strides = (1,1) # answer is [[23,33],[53,63]]
        thresholds = np.array([[23,32.9],[53,62.9]])
        expected_spikes = [1, 1]
        result = self.run_convolution_2d(thresholds)
        assert expected_spikes == self.calculated_spikes(thresholds,result)

        self.strides = (2,1) # answer is [[23,33]]
        thresholds = np.array([[22.9,33]])
        expected_spikes = [1]
        result = self.run_convolution_2d(thresholds)
        assert expected_spikes == self.calculated_spikes(thresholds,result)

        self.strides = (1,2) # answer is [[23],[53]]
        thresholds = np.array([[23],[52.9]])
        expected_spikes = [1]
        result = self.run_convolution_2d(thresholds)
        assert expected_spikes == self.calculated_spikes(thresholds,result)

        self.strides = (2,2) # answer is [[23]]
        thresholds = np.array([[22.9]])
        expected_spikes = [1]
        result = self.run_convolution_2d(thresholds)
        assert expected_spikes == self.calculated_spikes(thresholds,result)

    @pytest.mark.parametrize("basep", [3, 7])
    @pytest.mark.parametrize("bits", [3, 4, 7])
    @pytest.mark.parametrize("strides,nSpikes",
                             [
                                 ((1,1), 0),((1,1), 1),((1,1), 2),((1,1), 3),((1,1), 4),
                                 ((1,2), 0),((1,2), 1),((1,2), 2),
                                 ((2,1), 0),((2,1), 1),((2,1), 2),
                                 ((2,2), 0),((2,2), 1),
                             ])
    def test_valid_mode_with_strides(self,basep,bits,strides,nSpikes):
        self.basep = basep
        self.bits = bits
        self.mode = "valid"
        self.pvector = np.arange(1,10).reshape(3,3)
        self.pshape = self.pvector.shape
        self.strides = strides

        thresholds = get_spiked_thresholds(self.pvector,self.filters,self.strides,self.mode,nSpikes)
        result = self.run_convolution_2d(thresholds)

        assert self.expected_spikes(nSpikes) == self.calculated_spikes(thresholds,result)

    @pytest.mark.parametrize("mode,strides,nSpikes,biases",
                             [
                                 ("same",(1,1),4,None),("same",(1,1),4,0.),("same",(1,1),2,-17.),("same",(1,1),1,-20.),
                                 ("same",(1,2),2,-19.),("same",(1,2),1,-23),
                                 ("same",(2,1),0,-20.),("same",(2,1),2,-15.4),
                                 ("same",(2,2),0,-20.),("same",(2,2),1,-19.),
                                 ("valid",(1,1),0,-19.6),("valid",(1,1),1,-19),
                            ])
    def test_convolution_with_bias(self,mode,strides,nSpikes,biases):
        '''
            Convolution answer is [[20,16],[24,16]]
        '''
        image_height, image_width = 2, 2
        kernel_height, kernel_width = 2, 2
        nChannels = 1
        nFilters = 1

        self.basep = 3
        self.bits = 2
        self.pvector = generate_mock_image(image_height,image_width,nChannels).reshape(image_height,image_width)
        self.filters = generate_keras_kernel(kernel_height,kernel_width,nFilters,nChannels).reshape(kernel_height,kernel_width) # for now, this line ASSUMES nFilters and nChannels equals 1
        self.pshape = np.array(self.pvector).shape
        self.filters_shape = np.array(self.filters).shape
        self.mode = mode
        self.strides = strides
        self.biases = biases

        output_shape = get_output_shape(self.pvector,self.filters,self.strides,self.mode)
        thresholds = 0.5*np.ones(output_shape)
        result = self.run_convolution_2d(thresholds)

        assert self.expected_spikes(nSpikes) == self.calculated_spikes(thresholds,result)

    def test_explicit_convolution_with_bias(self):
        '''
        mode="full" 2D convolution result is [[1,4,4],[6,20,16],[9,24,16]].

        Keras 2d convolution brick does only supports mode="same" or mode="valid". The
        "same" result should return [[20,16],[24,16]], where padding is applied only to
        the bottom and right image data; padding is NOT appended to the left and top image positions.
        '''
        self.mode = "same"
        self.basep = 3 #basep
        self.bits = 3 #bits
        self.pvector = [[1, 2], [3, 4]]
        self.filters = [[1, 2], [3, 4]]
        self.pshape = np.array(self.pvector).shape
        self.filters_shape = np.array(self.filters).shape
        self.biases = -20.0

        # manually set strides, thresholds, and expected values
        self.strides = (1,1) # answer is [[20,16],[24,16]]
        thresholds = np.array([[0.5,0.5],[0.5,0.5]])
        expected_spikes = [1]
        result = self.run_convolution_2d(thresholds)
        assert expected_spikes == self.calculated_spikes(thresholds,result)

    @pytest.mark.xfail(reason="Not implemented.")
    def test_handling_of_4d_tensor_input(self):
        image_height, image_width = 3, 3
        kernel_height, kernel_width = 2, 2
        nChannels = 2
        nFilters = 3

        self.basep = 2
        self.bits = 2
        self.pvector = generate_mock_image(image_height,image_width,nChannels)
        self.filters = generate_keras_kernel(kernel_height,kernel_width,nFilters,nChannels)
        self.pshape = np.array(self.pvector).shape
        self.filters_shape = np.array(self.filters).shape
        self.mode = "same"
        self.strides = (1,1)
        assert False

    @pytest.mark.xfail(reason="Not implemented.")
    def test_4d_tensor_input(self):
        self.basep = 2
        self.bits = 2
        self.pvector = [[1, 1], [1, 1]]
        self.filters = [[1, 2], [3, 4]]
        self.pshape = np.array(self.pvector).shape
        self.filters_shape = np.array(self.filters).shape
        self.mode = "same"
        self.strides = (1,1)
        assert False

    # Auxillary/Helper Function below
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

    def run_convolution_2d(self, thresholds):
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
            keras_convolution_2d(
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
        scaffold.lay_bricks()
        scaffold.summary(verbose=1)
        backend = snn_Backend()
        backend_args = {}
        backend.compile(scaffold, backend_args)
        result = backend.run(5)
        return result

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

    @pytest.mark.parametrize("nSpikes,bias",[(0,-63),(1,-62),(2,-58)])
    def test_explicit_simple_convolution_with_bias(self,nSpikes,bias):
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
        result = self.run_convolution_2d(thresholds)
        assert self.expected_spikes(nSpikes) == self.calculated_spikes(thresholds,result)

    @pytest.mark.parametrize("nSpikes,biases",[(3,[-471, -1207, -1943]),(4,[-471, -1107, -1943]),(0,[-472, -1208, -1944])])
    def test_explicit_same_convolution_with_bias(self,nSpikes,biases):
        '''
            Given image and kernel provided in setup_method, the convolution ("same") answer is
                        Filter 1 Out            Filter 2 Out            Filter 3 Out
                  [[ 328.,  364.,  210.], [[ 808.,  908.,  498.], [[1288., 1452.,  786.], 
                   [ 436.,  472.,  270.],  [1108., 1208.,  654.],  [1780., 1944., 1038.], 
                   [ 299.,  321.,  180.]], [ 683.,  737.,  396.]], [1067., 1153.,  612.]])
        '''
        # TODO: Test different stride values
        self.biases = np.array(biases) # leads to 1 spike per filter output
        output_shape = keras_convolution2d_output_shape_4dinput(self.pvector,self.filters,self.strides,self.mode,self.nFilters)
        thresholds = 0.5*np.ones(output_shape).reshape(1,*output_shape)
        keras_convolution_answer = keras_convolve2d_4dinput(self.pvector,self.filters,strides=self.strides,mode=self.mode,filters=self.nFilters)
        result = self.run_convolution_2d(thresholds)
        assert self.expected_spikes(nSpikes) == self.calculated_spikes(thresholds,result)

    def test_explicit_3x3_simple_troubleshoot(self):
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
        nSpikes = 1
        biases = -62.0

        self.basep = 3
        self.bits = 3
        self.pvector = generate_mock_image(image_height,image_width,nChannels)
        self.filters = generate_keras_kernel(kernel_height,kernel_width,nFilters,nChannels)
        self.pshape = np.array(self.pvector).shape
        self.filters_shape = np.array(self.filters).shape
        self.nChannels = nChannels
        self.nFilters = nFilters
        self.biases = np.array([biases])

        output_shape = keras_convolution2d_output_shape_4dinput(self.pvector,self.filters,self.strides,self.mode,self.nFilters)
        thresholds = 0.5*np.ones(output_shape).reshape(1,*output_shape)
        keras_convolution_answer = keras_convolve2d_4dinput(self.pvector,self.filters,strides=self.strides,mode=self.mode,filters=self.nFilters)
        result = self.run_convolution_2d(thresholds)
        assert self.expected_spikes(nSpikes) == self.calculated_spikes(thresholds,result)
        assert False

    def test_explicit_2x2_simple_troubleshoot(self):
        '''
            Convolution answer is
                [[20., 16.],
                 [24., 16.]]
        '''
        image_height, image_width = 2, 2
        kernel_height, kernel_width = 2, 2
        nChannels = 1
        nFilters = 1
        nSpikes = 1
        biases = -23.0

        self.basep = 3
        self.bits = 3
        self.pvector = generate_mock_image(image_height,image_width,nChannels)
        self.filters = generate_keras_kernel(kernel_height,kernel_width,nFilters,nChannels)
        self.pshape = np.array(self.pvector).shape
        self.filters_shape = np.array(self.filters).shape
        self.nChannels = nChannels
        self.nFilters = nFilters
        self.biases = np.array([biases])

        output_shape = keras_convolution2d_output_shape_4dinput(self.pvector,self.filters,self.strides,self.mode,self.nFilters)
        thresholds = 0.5*np.ones(output_shape).reshape(1,*output_shape)
        keras_convolution_answer = keras_convolve2d_4dinput(self.pvector,self.filters,strides=self.strides,mode=self.mode,filters=self.nFilters)
        result = self.run_convolution_2d(thresholds)
        assert self.expected_spikes(nSpikes) == self.calculated_spikes(thresholds,result)
        assert False

    @pytest.mark.parametrize("nSpikes,biases,strides",[(16,[-320, -682, -1152],(1,2)),(13,[-298, -682, -1037],(2,1)),(0,[-472, -1208, -1944],(1,2)),(3,[-436, -1108, -1780],(2,1)),(0,[-328, -1208, -1944],(2,2))])
    def test_explicit_same_convolution_with_bias_and_strides(self,nSpikes,biases,strides):
        '''
            Given image and kernel provided in setup_method, the convolution ("same") answer is
                        Filter 1 Out            Filter 2 Out            Filter 3 Out
                  [[ 328.,  364.,  210.], [[ 808.,  908.,  498.], [[1288., 1452.,  786.],
                   [ 436.,  472.,  270.],  [1108., 1208.,  654.],  [1780., 1944., 1038.],
                   [ 299.,  321.,  180.]], [ 683.,  737.,  396.]], [1067., 1153.,  612.]])
        '''
        # TODO: Test different stride values
        self.strides = strides
        self.biases = np.array(biases) # leads to 1 spike per filter output
        output_shape = keras_convolution2d_output_shape_4dinput(self.pvector,self.filters,self.strides,self.mode,self.nFilters)
        thresholds = 0.5*np.ones(output_shape).reshape(1,*output_shape)
        keras_convolution_answer = keras_convolve2d_4dinput(self.pvector,self.filters,strides=self.strides,mode=self.mode,filters=self.nFilters)
        result = self.run_convolution_2d(thresholds)
        assert self.expected_spikes(nSpikes) == self.calculated_spikes(thresholds,result)

    @pytest.mark.parametrize("nSpikes,biases",[(3,[-471, -1207, -1943]),(4,[-471, -1107, -1943]),(0,[-472, -1208, -1944]),(2,[-435, -1208, -1944]),(3,[-472, -907, -1944]),(2,[-472, -1208, -1779])])
    def test_explicit_valid_convolution_with_bias(self,nSpikes,biases):
        '''
            Given image and kernel provided in setup_method, the convolution ("valid") answer is
                        Filter 1 Out            Filter 2 Out            Filter 3 Out
                  [[ 328.,  364.],        [[ 808.,  908.],        [[1288., 1452.],
                   [ 436.,  472.]],        [1108., 1208.]],        [1780., 1944.]])
        '''
        self.mode = "valid"
        self.biases = np.array(biases) # leads to 1 spike per filter output
        output_shape = keras_convolution2d_output_shape_4dinput(self.pvector,self.filters,self.strides,self.mode,self.nFilters)
        thresholds = 0.5*np.ones(output_shape).reshape(1,*output_shape)
        keras_convolution_answer = keras_convolve2d_4dinput(self.pvector,self.filters,strides=self.strides,mode=self.mode,filters=self.nFilters)
        result = self.run_convolution_2d(thresholds)
        assert self.expected_spikes(nSpikes) == self.calculated_spikes(thresholds,result)

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
        self.biases = np.array(biases) # leads to 1 spike per filter output
        output_shape = keras_convolution2d_output_shape_4dinput(self.pvector,self.filters,self.strides,self.mode,self.nFilters)
        thresholds = 0.5*np.ones(output_shape).reshape(1,*output_shape)
        keras_convolution_answer = keras_convolve2d_4dinput(self.pvector,self.filters,strides=self.strides,mode=self.mode,filters=self.nFilters)
        result = self.run_convolution_2d(thresholds)
        assert self.expected_spikes(nSpikes) == self.calculated_spikes(thresholds,result)

    @pytest.mark.xfail(reason="Not implemented.")
    def test_handling_of_4d_tensor_input(self):
        assert False

    @pytest.mark.xfail(reason="Not implemented.")
    def test_4d_tensor_input(self):
        assert False

    # Auxillary/Helper Function below
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

    def run_convolution_2d(self, thresholds):
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
        scaffold.lay_bricks()
        scaffold.summary(verbose=1)
        backend = snn_Backend()
        backend_args = {}
        backend.compile(scaffold, backend_args)
        result = backend.run(5)
        return result