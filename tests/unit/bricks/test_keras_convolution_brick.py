import numpy as np
import pytest
import logging
from contextlib import nullcontext as does_not_raise

from fugu.backends import snn_Backend
from fugu.bricks.keras_convolution_bricks import keras_convolution_2d as convolution_2d
from fugu.bricks.input_bricks import BaseP_Input
from fugu.scaffold import Scaffold
from fugu.utils.keras_helpers import keras_convolve2d, keras_convolve2d_4dinput, generate_keras_kernel, generate_mock_image

from scipy.signal import convolve2d

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
        layer = convolution_2d(self.pshape,self.filters,np.ones(self.filters_shape),self.basep,self.bits,name='conv',strides=self.strides)
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
            layer = convolution_2d(self.pshape,self.filters,np.ones(self.filters_shape),self.basep,self.bits,name='conv',strides=self.strides)
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

        thresholds = get_unique_thresholds(self.pvector,self.filters,self.strides,self.mode,nSpikes)
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
    def test_valid_mode_with_strides(self,basep,bits,strides,nSpikes):
        self.basep = basep
        self.bits = bits
        self.mode = "valid"
        self.pvector = np.arange(1,10).reshape(3,3)
        self.pshape = self.pvector.shape
        self.strides = strides

        thresholds = get_unique_thresholds(self.pvector,self.filters,self.strides,self.mode,nSpikes)
        result = self.run_convolution_2d(thresholds)

        assert self.expected_spikes(nSpikes) == self.calculated_spikes(thresholds,result)

    @pytest.mark.xfail(reason="Not implemented.")
    def test_handling_of_4d_tensor_input(self):
        self.basep = 2
        self.bits = 2
        self.pvector = generate_mock_image(2,2,2)
        self.filters = generate_keras_kernel(2,2,3,2)
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
            convolution_2d(
                self.pshape,
                self.filters,
                thresholds,
                self.basep,
                self.bits,
                name="convolution_",
                mode=self.mode,
                strides=self.strides,
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

def get_unique_thresholds(image,kernel,strides,mode,nSpikes):
    from scipy.signal import convolve2d
    # image = self.pvector
    # kernel = self.filters
    Srow, Scol = strides
    mode_answer = np.flip(convolve2d(np.flip(image),np.flip(kernel),mode=mode))
    strided_answer = mode_answer[::Srow,::Scol]

    subt = np.zeros(np.size(strided_answer))
    subt[:nSpikes] = 0.1
    subt = np.reshape(subt, strided_answer.shape)
    thresholds = strided_answer - subt

    return thresholds
