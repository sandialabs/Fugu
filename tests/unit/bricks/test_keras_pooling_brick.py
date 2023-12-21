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

        self.basep = 3
        self.bits = 3
        self.filters = [[2, 3],[4,5]]
        self.mock_input = generate_mock_image(image_height,image_width,nChannels=nChannels)
        self.input_shape = self.mock_input.shape
        self.data_format = "channels_last"

    @pytest.fixture
    def default_pooling_params(self, default_convolution_params):
        pool_height, pool_width = 2, 2

        self.pool_size = (pool_height, pool_width)
        self.pool_padding = "same"
        self.pool_method = "max"
        self.pool_strides = (1,1)
        self.pool_output_shape = get_pool_output_shape(self.convolution_input_shape,self.pool_size,self.pool_strides,self.pool_padding,data_format=self.data_format)
        self.pool_thresholds = 0.9*np.ones(self.pool_output_shape)
        self.pool_input_shape = self.convolution_output_shape

    @pytest.fixture
    def convolution_mode_fixture(self):
        return "same"

    @pytest.fixture
    def default_convolution_params(self, convolution_mode_fixture):
        image_height, image_width, nChannels = 3, 3, 2
        kernel_height, kernel_width, nFilters = 2, 2, 3

        self.basep = 3
        self.bits = 4
        self.convolution_strides = (1,1)
        self.convolution_mode = convolution_mode_fixture
        self.convolution_mock_image = generate_mock_image(image_height, image_width, nChannels=nChannels)
        self.convolution_input_shape = self.convolution_mock_image.shape
        self.convolution_filters = generate_keras_kernel(kernel_height,kernel_width,nFilters,nChannels)
        self.convolution_thresholds = 0.5
        self.convolution_biases = np.array([-471., -1207., -1943.])
        self.convolution_output_shape = (1,*keras_convolution2d_output_shape_4dinput(self.convolution_mock_image,self.convolution_filters,self.convolution_strides,self.convolution_mode,nFilters=nFilters))
        self.convolution_answer = keras_convolve2d_4dinput(self.convolution_mock_image, self.convolution_filters, self.convolution_strides, self.convolution_mode, self.data_format,filters=nFilters).reshape(self.convolution_output_shape)

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
    def test_pooling_size_input(self, default_pooling_params, pooling_size, expectation):
        self.pool_size = pooling_size
        self.pool_thresholds = 0.9
        with expectation:
            self.run_pooling_2d()

    @pytest.mark.parametrize("pool_strides,expectation", [(None,does_not_raise()),(2,does_not_raise()),((2,2),does_not_raise()),(2.0,pytest.raises(ValueError)),((2,2,1),pytest.raises(ValueError)),([2,2],pytest.raises(ValueError))])
    def test_pooling_strides_input(self, default_pooling_params, pool_strides, expectation):
        self.pool_strides = pool_strides
        self.pool_thresholds = 0.9
        with expectation:
            self.run_pooling_2d()

    def test_explicit_max_pooling_same_mode_strides_11(self, default_pooling_params):
        self.pool_strides = (1,1)
        self.pool_thresholds = 0.9

        self.convolution_answer_boolean = self.convolution_answer + self.convolution_biases > 0.5
        expected_pool_answer = self.get_expected_pooling_answer(self.convolution_answer_boolean)
        result = self.run_pooling_2d()

        assert False

    @pytest.mark.xfail(reason="Not implemented.")
    def test_average_pooling(self, default_pooling_params):
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

    def run_pooling_2d(self):
        scaffold = Scaffold()
        scaffold.add_brick(BaseP_Input(self.mock_input,p=self.basep,bits=self.bits,collapse_binary=False,name="I",time_dimension=False),"input")
        scaffold.add_brick(keras_convolution_2d(self.convolution_input_shape,self.convolution_filters,self.convolution_thresholds,self.basep,self.bits,name="convolution_",mode=self.convolution_mode,strides=self.convolution_strides,biases=self.convolution_biases),[(0, 0)],output=True)
        scaffold.add_brick(keras_pooling_2d(self.pool_size,self.pool_strides,thresholds=self.pool_thresholds,name="pool_",padding=self.pool_padding,method=self.pool_method),[(1,0)],output=True)

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
    
    def get_expected_pooling_answer(self, pool_input):
        batch_size, image_height, image_width, nChannels = get_pool_input_shape_params(self.convolution_output_shape, self.data_format)
        spatial_input_shape = get_spatial_input_shape(self.convolution_output_shape,self.data_format)
        row_stride_positions, col_stride_positions = get_stride_positions(spatial_input_shape, self.pool_strides)
        spatial_output_shape = get_spatial_output_shape(self.convolution_input_shape,self.data_format,self.pool_size,self.pool_padding,self.pool_strides)
        output_shape = get_pool_output_shape(self.convolution_output_shape,self.pool_size, self.pool_strides, self.pool_padding, self.data_format)
        expected = []


        if self.pool_method == "max":
            for row in row_stride_positions[:spatial_output_shape[0]]:
                for col in col_stride_positions[:spatial_output_shape[1]]:
                    for channel in np.arange(nChannels):
                        expected.append(np.any(pool_input[0,row:row+self.pool_size[0],col:col+self.pool_size[1],channel]))

            expected = np.reshape(expected, output_shape).astype(int)
            expected = (np.array(expected, dtype=int) > 0.9).astype(float)

        elif self.pool_method == "average": #TODO : fix this code. 
            weights = np.ones(output_shape,dtype=float) / np.prod(self.pool_size)
            for row in row_stride_positions[:spatial_output_shape[0]]:
                for col in col_stride_positions[:spatial_output_shape[1]]:
                    for channel in np.arange(nChannels):
                        expected.append(np.dot(weights.flatten(),pool_input[0,row:row+self.pool_size[0],col:col+self.pool_size[1],channel].astype(int).flatten()))

            expected = np.reshape(expected, output_shape).astype(float)
        else:
            print(f"'method' class member variable must be either 'max' or 'average'. But it is {self.method}.")
            raise ValueError("Unrecognized 'method' class member variable.")
            
        return np.array(expected)
    
    def get_expected_spikes(self,expected_ans):
        ans = np.array(expected_ans)
        spikes = ans[ ans > self.thresholds].astype(float)
        return list(2.0 * np.ones(spikes.shape))    

