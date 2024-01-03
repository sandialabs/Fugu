# isort: skip_file
# fmt: off
import numpy as np
import pytest
from contextlib import nullcontext as does_not_raise

from fugu.backends import snn_Backend
from fugu.bricks.keras_convolution_bricks import keras_convolution_2d_4dinput as keras_convolution_2d
from fugu.bricks.input_bricks import BaseP_Input
from fugu.bricks.keras_pooling_bricks import keras_pooling_2d_4dinput as keras_pooling_2d
from fugu.scaffold import Scaffold

from fugu.utils.keras_helpers import keras_convolve2d, keras_convolve2d_4dinput, generate_keras_kernel, generate_mock_image, keras_convolution2d_output_shape_4dinput
from ..helpers import ConvolutionParams, PoolingParams

def get_pool_input_shape_params(input_shape,data_format):
    if data_format.lower() == "channels_last":
        batch_size, image_height, image_width, nChannels = input_shape
    elif data_format.lower() == "channels_first":
        batch_size, nChannels, image_height, image_width = input_shape
    else:
        raise ValueError(f"'data_format' is either 'channels_first' or 'channels_last'. Received {data_format}")

    return batch_size, image_height, image_width, nChannels

def get_pool_output_shape(input_shape, pool_size, pool_strides, pool_padding, data_format="channels_last"):
    if pool_strides is None:
        pool_strides = pool_size

    batch_size, image_height, image_width, nChannels = get_pool_input_shape_params(input_shape,data_format)
    spatial_input_shape = (image_height, image_width)

    spatial_output_shape = get_padding_output_shape(spatial_input_shape,pool_size,pool_strides,pool_padding)

    if data_format.lower() == "channels_last":
        output_shape = (batch_size, *spatial_output_shape, nChannels)
    else:
        output_shape = (batch_size, nChannels, *spatial_output_shape)

    return output_shape

def get_spatial_input_shape(input_shape,data_format):
    batch_size, image_height, image_width, nChannels = get_pool_input_shape_params(input_shape,data_format)
    spatial_input_shape = (image_height, image_width)
    return spatial_input_shape

def get_spatial_output_shape(input_shape,data_format,pool_size,pool_padding,pool_strides):
    spatial_input_shape = get_spatial_input_shape(input_shape,data_format)
    if pool_padding.lower() == "same":
        spatial_output_shape = same_padding_spatial_output_shape(spatial_input_shape,pool_strides)
    elif pool_padding.lower() == "valid":
        spatial_output_shape = valid_padding_spatial_output_shape(spatial_input_shape,pool_size,pool_strides)
    else:
        raise ValueError(f"'pool_padding' is one of 'same' or 'valid'. Received {pool_padding}.")

    spatial_output_shape = list(map(int,spatial_output_shape))
    return spatial_output_shape
    
def same_padding_spatial_output_shape(spatial_input_shape,  pool_strides):
    return np.floor((np.array(spatial_input_shape) - 1) / np.array(pool_strides)) + 1

def valid_padding_spatial_output_shape(spatial_input_shape, pool_size, pool_strides):
    return np.floor((np.array(spatial_input_shape) - np.array(pool_size)) / np.array(pool_strides)) + 1

def get_padding_output_shape(spatial_input_shape,pool_size,pool_strides,pool_padding):
    if pool_strides is None:
        pool_strides = pool_size

    if pool_padding.lower() == "same":
        spatial_output_shape = same_padding_spatial_output_shape(spatial_input_shape,pool_strides)
    elif pool_padding.lower() == "valid":
        spatial_output_shape = valid_padding_spatial_output_shape(spatial_input_shape,pool_size,pool_strides)
    else:
        raise ValueError(f"'pool_padding' is one of 'same' or 'valid'. Received {pool_padding}.")

    spatial_output_shape = list(map(int,spatial_output_shape))
    return spatial_output_shape

def stride_positions(pixel_dim, stride_len):
    return np.arange(0, pixel_dim, stride_len, dtype=int)

def get_stride_positions(spatial_input_shape, strides):
    return [stride_positions(spatial_input_shape[0],strides[0]), stride_positions(spatial_input_shape[1],strides[1])]
    
class Test_KerasPooling2D:

    def setup_method(self):
        image_height, image_width, nChannels = 3, 3, 2
        kernel_height, kernel_width, nFilters = 2, 2, 3
        pool_height, pool_width = 2, 2

        self.basep = 4
        self.bits = 4
        self.filters = [[2, 3],[4,5]]
        self.mock_input = generate_mock_image(image_height,image_width,nChannels=nChannels)
        self.input_shape = self.mock_input.shape
        self.data_format = "channels_last"

    @pytest.fixture
    def convolution_mode_fixture(self):
        return "same"

    @pytest.fixture(params=["fixed"])
    def spike_positions_vector(self, request):
        return request.param
    
    @pytest.fixture
    def pooling_size(self):
        return 2
    
    @pytest.fixture
    def pooling_stride(self):
        return 2
    
    @pytest.fixture
    def pooling_method(self):
        return "max"

    @pytest.mark.parametrize("pooling_size,expectation", [(2,does_not_raise()),((2,2),does_not_raise()),(2.0,pytest.raises(ValueError)),((2,2,1),pytest.raises(ValueError)),([2,2],pytest.raises(ValueError))])
    def test_pooling_size_input(self, pooling_size, expectation):
        convo_obj = ConvolutionParams(biases=np.array([-471., -1207., -1943.]))
        pool_obj = PoolingParams(convo_obj)
        pool_obj.pool_size = pooling_size
        with expectation:
            self.run_pooling_2d(convo_obj,pool_obj)

    @pytest.mark.parametrize("pool_strides,expectation", [(None,does_not_raise()),(2,does_not_raise()),((2,2),does_not_raise()),(2.0,pytest.raises(ValueError)),((2,2,1),pytest.raises(ValueError)),([2,2],pytest.raises(ValueError))])
    def test_pooling_strides_input(self, pool_strides, expectation):
        convo_obj = ConvolutionParams(biases=np.array([-471., -1207., -1943.]))
        pool_obj = PoolingParams(convo_obj)
        pool_obj.pool_strides = pool_strides
        with expectation:
            self.run_pooling_2d(convo_obj,pool_obj)

    def test_explicit_max_pooling_same_mode_strides_11(self):
        convo_obj = ConvolutionParams(biases=np.array([-471., -1207., -1943.]))
        pool_obj = PoolingParams(convo_obj, pool_strides=(1,1), pool_padding="same")

        expected_spike_count = (pool_obj.pool_answer > pool_obj.pool_thresholds).sum().astype(int)

        result = self.run_pooling_2d(convo_obj,pool_obj)
        calculated_spike_count = len(result[result['time'] > 1].index)
        assert expected_spike_count == calculated_spike_count

    def test_explicit_max_pooling_same_mode_strides_12(self):
        convo_obj = ConvolutionParams(biases=np.array([-471., -1207., -1943.]))
        pool_obj = PoolingParams(convo_obj, pool_strides=(1,2), pool_padding="same")

        expected_spike_count = (pool_obj.pool_answer > pool_obj.pool_thresholds).sum().astype(int)

        result = self.run_pooling_2d(convo_obj,pool_obj)
        calculated_spike_count = len(result[result['time'] > 1].index)
        assert expected_spike_count == calculated_spike_count

    def test_explicit_max_pooling_same_mode_strides_21(self):
        convo_obj = ConvolutionParams(biases=np.array([-471., -1207., -1943.]))
        pool_obj = PoolingParams(convo_obj, pool_strides=(2,1), pool_padding="same")

        expected_spike_count = (pool_obj.pool_answer > pool_obj.pool_thresholds).sum().astype(int)

        result = self.run_pooling_2d(convo_obj,pool_obj)
        calculated_spike_count = len(result[result['time'] > 1].index)
        assert expected_spike_count == calculated_spike_count

    def test_explicit_max_pooling_same_mode_strides_22(self):
        convo_obj = ConvolutionParams(biases=np.array([-471., -1207., -1943.]))
        pool_obj = PoolingParams(convo_obj, pool_strides=(2,2), pool_padding="same")

        expected_spike_count = (pool_obj.pool_answer > pool_obj.pool_thresholds).sum().astype(int)

        result = self.run_pooling_2d(convo_obj,pool_obj)
        calculated_spike_count = len(result[result['time'] > 1].index)
        assert expected_spike_count == calculated_spike_count

    def test_explicit_max_pooling_valid_mode_strides_11(self):
        convo_obj = ConvolutionParams(biases=np.array([-471., -1207., -1943.]))
        pool_obj = PoolingParams(convo_obj, pool_strides=(1,1), pool_padding="valid")

        expected_spike_count = (pool_obj.pool_answer > pool_obj.pool_thresholds).sum().astype(int)

        result = self.run_pooling_2d(convo_obj,pool_obj)
        calculated_spike_count = len(result[result['time'] > 1].index)
        assert expected_spike_count == calculated_spike_count

    def test_explicit_max_pooling_valid_mode_strides_12(self):
        convo_obj = ConvolutionParams(biases=np.array([-471., -1207., -1943.]))
        pool_obj = PoolingParams(convo_obj, pool_strides=(1,2), pool_padding="valid")

        expected_spike_count = (pool_obj.pool_answer > pool_obj.pool_thresholds).sum().astype(int)

        result = self.run_pooling_2d(convo_obj,pool_obj)
        calculated_spike_count = len(result[result['time'] > 1].index)
        assert expected_spike_count == calculated_spike_count

    def test_explicit_max_pooling_valid_mode_strides_21(self):
        convo_obj = ConvolutionParams(biases=np.array([-471., -1207., -1943.]))
        pool_obj = PoolingParams(convo_obj, pool_strides=(2,1), pool_padding="valid")

        expected_spike_count = (pool_obj.pool_answer > pool_obj.pool_thresholds).sum().astype(int)

        result = self.run_pooling_2d(convo_obj,pool_obj)
        calculated_spike_count = len(result[result['time'] > 1].index)
        assert expected_spike_count == calculated_spike_count

    def test_explicit_max_pooling_valid_mode_strides_22(self):
        convo_obj = ConvolutionParams(biases=np.array([-471., -1207., -1943.]))
        pool_obj = PoolingParams(convo_obj, pool_strides=(2,2), pool_padding="valid")

        expected_spike_count = (pool_obj.pool_answer > pool_obj.pool_thresholds).sum().astype(int)

        result = self.run_pooling_2d(convo_obj,pool_obj)
        calculated_spike_count = len(result[result['time'] > 1].index)
        assert expected_spike_count == calculated_spike_count

    def test_explicit_average_pooling_same_mode_strides_11(self):
        convo_obj = ConvolutionParams(biases=np.array([-471., -1207., -1943.]))
        pool_obj = PoolingParams(convo_obj, pool_method="average", pool_strides=(1,1), pool_padding="same")

        expected_spike_count = (pool_obj.pool_answer > pool_obj.pool_thresholds).sum().astype(int)

        result = self.run_pooling_2d(convo_obj,pool_obj)
        calculated_spike_count = len(result[result['time'] > 1].index)
        assert expected_spike_count == calculated_spike_count

    def test_explicit_average_pooling_same_mode_strides_12(self):
        convo_obj = ConvolutionParams(biases=np.array([-471., -1207., -1943.]))
        pool_obj = PoolingParams(convo_obj, pool_method="average", pool_strides=(1,2), pool_padding="same")

        expected_spike_count = (pool_obj.pool_answer > pool_obj.pool_thresholds).sum().astype(int)

        result = self.run_pooling_2d(convo_obj,pool_obj)
        calculated_spike_count = len(result[result['time'] > 1].index)
        assert expected_spike_count == calculated_spike_count

    def test_explicit_average_pooling_same_mode_strides_21(self):
        convo_obj = ConvolutionParams(biases=np.array([-471., -1207., -1943.]))
        pool_obj = PoolingParams(convo_obj, pool_method="average", pool_strides=(2,1), pool_padding="same")

        expected_spike_count = (pool_obj.pool_answer > pool_obj.pool_thresholds).sum().astype(int)

        result = self.run_pooling_2d(convo_obj,pool_obj)
        calculated_spike_count = len(result[result['time'] > 1].index)
        assert expected_spike_count == calculated_spike_count

    def test_explicit_average_pooling_same_mode_strides_22(self):
        convo_obj = ConvolutionParams(biases=np.array([-471., -1207., -1943.]))
        pool_obj = PoolingParams(convo_obj, pool_method="average", pool_strides=(2,2), pool_padding="same")

        expected_spike_count = (pool_obj.pool_answer > pool_obj.pool_thresholds).sum().astype(int)

        result = self.run_pooling_2d(convo_obj,pool_obj)
        calculated_spike_count = len(result[result['time'] > 1].index)
        assert expected_spike_count == calculated_spike_count

    def test_explicit_average_pooling_valid_mode_strides_11(self):
        convo_obj = ConvolutionParams(biases=np.array([-471., -1207., -1943.]))
        pool_obj = PoolingParams(convo_obj, pool_method="average", pool_strides=(1,1), pool_padding="valid")

        expected_spike_count = (pool_obj.pool_answer > pool_obj.pool_thresholds).sum().astype(int)

        result = self.run_pooling_2d(convo_obj,pool_obj)
        calculated_spike_count = len(result[result['time'] > 1].index)
        assert expected_spike_count == calculated_spike_count

    def test_explicit_average_pooling_valid_mode_strides_12(self):
        convo_obj = ConvolutionParams(biases=np.array([-471., -1207., -1943.]))
        pool_obj = PoolingParams(convo_obj, pool_method="average", pool_strides=(1,2), pool_padding="valid")

        expected_spike_count = (pool_obj.pool_answer > pool_obj.pool_thresholds).sum().astype(int)

        result = self.run_pooling_2d(convo_obj,pool_obj)
        calculated_spike_count = len(result[result['time'] > 1].index)
        assert expected_spike_count == calculated_spike_count

    def test_explicit_average_pooling_valid_mode_strides_21(self):
        convo_obj = ConvolutionParams(biases=np.array([-471., -1207., -1943.]))
        pool_obj = PoolingParams(convo_obj, pool_method="average", pool_strides=(2,1), pool_padding="valid")

        expected_spike_count = (pool_obj.pool_answer > pool_obj.pool_thresholds).sum().astype(int)

        result = self.run_pooling_2d(convo_obj,pool_obj)
        calculated_spike_count = len(result[result['time'] > 1].index)
        assert expected_spike_count == calculated_spike_count

    def test_explicit_average_pooling_valid_mode_strides_22(self):
        convo_obj = ConvolutionParams(biases=np.array([-471., -1207., -1943.]))
        pool_obj = PoolingParams(convo_obj, pool_method="average", pool_strides=(2,2), pool_padding="valid")

        expected_spike_count = (pool_obj.pool_answer > pool_obj.pool_thresholds).sum().astype(int)

        result = self.run_pooling_2d(convo_obj,pool_obj)
        calculated_spike_count = len(result[result['time'] > 1].index)
        assert expected_spike_count == calculated_spike_count

    @pytest.mark.parametrize("pool_size", [(1,2),(2,1),(2,2),(2,3),(3,2),(3,3)])
    @pytest.mark.parametrize("pool_strides", [(1,1),(1,2),(2,1),(2,2),(1,3),(3,1),(2,3),(3,2),(3,3)])
    @pytest.mark.parametrize("pool_padding", ["same", "valid"])
    @pytest.mark.parametrize("pool_method", ["max", "average"])
    @pytest.mark.parametrize("nChannels,nFilters", [(2,3), (1,1)])
    def test_pooling_exhaustively_with_random_biases(self, pool_size, pool_strides, pool_padding, pool_method, nChannels, nFilters):
        self.basep = 4
        self.bits = 3

        convo_obj = ConvolutionParams(image_height=5, image_width=5, nChannels=nChannels, kernel_height=2, kernel_width=2, nFilters=nFilters, biases=None)
        convo_obj.biases = convo_obj.get_random_biases_within_answer_range()
        convo_obj._set_convolution_answer_boolean()

        pool_obj = PoolingParams(convo_obj, pool_size=pool_size, pool_strides=pool_strides, pool_padding=pool_padding, pool_method=pool_method)
        expected_spike_count = (pool_obj.pool_answer > pool_obj.pool_thresholds).sum().astype(int)

        result = self.run_pooling_2d(convo_obj,pool_obj)
        calculated_spike_count = len(result[result['time'] > 1].index)
        assert expected_spike_count == calculated_spike_count

    def test_pooling_one_off(self):
        self.basep = 4
        self.bits = 3

        convo_obj = ConvolutionParams(image_height=5, image_width=5, nChannels=2, kernel_height=2, kernel_width=2, nFilters=3, biases=None)
        convo_obj.biases = convo_obj.get_random_biases_within_answer_range()
        convo_obj._set_convolution_answer_boolean()

        pool_obj = PoolingParams(convo_obj, pool_size=(2,3), pool_strides=(1,1), pool_padding="same", pool_method="max")
        expected_spike_count = (pool_obj.pool_answer > pool_obj.pool_thresholds).sum().astype(int)

        result = self.run_pooling_2d(convo_obj,pool_obj)
        calculated_spike_count = len(result[result['time'] > 1].index)
        assert expected_spike_count == calculated_spike_count

    @pytest.mark.xfail(reason="Not implemented.")
    def test_data_format_channels_last(self):
        assert False

    def get_pool_output_shape(self):
        return np.floor((self.input_shape - 1) / self.pool_strides) + 1

    def get_output_neuron_numbers(self):
        neuron_numbers = []
        for key in self.graph.nodes.keys():
            if key.startswith('pool_p'):
                neuron_numbers.append(self.graph.nodes[key]['neuron_number'])

        return np.array(neuron_numbers)

    def get_output_spike_positions(self):
        neuron_numbers = self.get_output_neuron_numbers()
        if neuron_numbers.size == 0:
            output_ini_position = np.nan
            output_end_position = np.nan
        else:
            output_ini_position = np.amin(neuron_numbers)
            output_end_position = np.amax(neuron_numbers)
        return [output_ini_position,output_end_position]

    def get_output_mask(self,output_positions, result):
        ini = output_positions[0]
        end = output_positions[1]
        return (result["neuron_number"] >= ini) & (result["neuron_number"] <= end)

    def run_pooling_2d(self,convo_obj, pool_obj):
        scaffold = Scaffold()
        scaffold.add_brick(BaseP_Input(convo_obj.mock_image,p=self.basep,bits=self.bits,collapse_binary=False,name="I",time_dimension=False),"input")
        scaffold.add_brick(keras_convolution_2d(convo_obj.input_shape,convo_obj.filters,convo_obj.thresholds,self.basep,self.bits,name="convolution_",mode=convo_obj.mode,strides=convo_obj.strides,biases=convo_obj.biases),[(0, 0)],output=True)
        scaffold.add_brick(keras_pooling_2d(pool_obj.pool_size,pool_obj.pool_strides,thresholds=pool_obj.pool_thresholds,name="pool_",padding=pool_obj.pool_padding,method=pool_obj.pool_method),[(1,0)],output=True)

        self.graph = scaffold.lay_bricks()
        scaffold.summary(verbose=1)
        backend = snn_Backend()
        backend_args = {}
        backend.compile(scaffold, backend_args)
        result = backend.run(5)
        return result

    def make_random_thresholds_vector(self, initial_vector):
        thresholds = initial_vector.copy()
        random_ids = np.random.randint(2,size=thresholds.shape).astype(bool)
        thresholds[random_ids] = thresholds[random_ids] - 0.1
        thresholds[~random_ids] = thresholds[~random_ids] + 0.1
        thresholds[thresholds < 0.0] = 0.0
        return thresholds
    
    def make_convolution_spike_positions_vector(self, case):
        '''
        The returned list is [0.0, 0.1, 0.0, 0.1, 0.0, 0.0, 0.0] for "fixed" case. The "random" case returns a list 
        with the positions of "0.1" randomized in the list.

        The 0.1 specifies the convolution brick's output positions that will spike (i.e., output neurons will fire).
        '''
        conv_spike_pos = np.zeros(self.conv_ans.shape, dtype=float)

        if case.lower() == "fixed":
            conv_spike_pos = 0.1*np.array([[1,0,0,0,1],[0,0,1,1,1],[1,1,1,0,1],[0,1,0,0,1],[1,1,0,0,0]],dtype=float)
        elif case.lower() == "random":
            conv_spike_pos = 0.1 * np.random.randint(2,size=self.conv_ans.shape)
        else:
            print(f"Case parameter is either 'fixed' or 'random'. But you provided {case}.")
            raise ValueError("Unrecognized 'case' parameter value provided.")
        
        return conv_spike_pos
    
    def get_expected_pooling_answer(self, pool_input, pool_obj):
        row_stride_positions, col_stride_positions = get_stride_positions(pool_obj.spatial_input_shape, pool_obj.pool_strides)
        expected = []

        if pool_obj.pool_method == "max":
            for row in row_stride_positions[:pool_obj.spatial_output_shape[0]]:
                for col in col_stride_positions[:pool_obj.spatial_output_shape[1]]:
                    for channel in np.arange(pool_obj.nChannels):
                        expected.append(np.any(pool_input[0,row:row+pool_obj.pool_size[0],col:col+pool_obj.pool_size[1],channel]))

            expected = np.reshape(expected, pool_obj.output_shape).astype(int)
            expected = (np.array(expected, dtype=int) > 0.9).astype(float)

        elif pool_obj.pool_method == "average":
            weights = 1.0 / np.prod(pool_obj.pool_size)
            for row in row_stride_positions[:pool_obj.spatial_output_shape[0]]:
                for col in col_stride_positions[:pool_obj.spatial_output_shape[1]]:
                    for channel in np.arange(pool_obj.nChannels):
                        expected.append( (weights * pool_input[0,row:row+pool_obj.pool_size[0],col:col+pool_obj.pool_size[1],channel].astype(int)).sum() )

            expected = np.reshape(expected, pool_obj.output_shape).astype(float)
        else:
            print(f"'method' class member variable must be either 'max' or 'average'. But it is {pool_obj.pool_method}.")
            raise ValueError("Unrecognized 'method' class member variable.")
            
        return np.array(expected)
    
    def get_expected_spikes(self,expected_ans):
        ans = np.array(expected_ans)
        spikes = ans[ ans > self.thresholds].astype(float)
        return list(2.0 * np.ones(spikes.shape))    

