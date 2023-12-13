# isort: skip_file
import numpy as np
import pytest

from fugu.backends import snn_Backend
from fugu.bricks.convolution_bricks import convolution_1d, convolution_2d
from fugu.bricks.input_bricks import BaseP_Input
from fugu.bricks.pooling_bricks import pooling_1d, pooling_2d
from fugu.scaffold import Scaffold

# Turn off black formatting for this file
# fmt: off

convolve2d = pytest.importorskip("scipy.signal", reason=f"Scipy package not installed. Skipping test file {__file__} because of module dependency.").convolve2d

@pytest.mark.skip(reason="All test need to be updated for the keras pooling brick.")
class Test_KerasPooling2D:

    @pytest.fixture
    def convolution_mode(self):
        return "same"

    @pytest.fixture
    def default_convolution_params(self, convolution_mode):
        self.basep = 3
        self.bits = 3
        self.filters = [[2, 3],[4,5]]
        self.mode = convolution_mode
        self.pvector = [[1,1,4,6],[6,2,4,2],[2,3,5,4],[6,1,6,3]]
        self.pshape = np.array(self.pvector).shape

    @pytest.fixture
    def numpy_convolution_result(self, default_convolution_params, spike_positions_vector):
        self.conv_ans = convolve2d(self.pvector, self.filters, mode=self.mode)
        self.pixel_shape = self.conv_ans.shape
        self.pixel_dim1 = self.pixel_shape[0]
        self.pixel_dim2 = self.pixel_shape[1]
        self.conv_spike_pos = self.make_convolution_spike_positions_vector(spike_positions_vector)
        self.conv_thresholds = self.conv_ans - self.conv_spike_pos
        return self.conv_ans > self.conv_thresholds        

    @pytest.fixture(params=["fixed", "random"])
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
    
    @pytest.fixture
    def default_pooling_params(self, pooling_size, pooling_stride, pooling_method):
        self.pool_size = pooling_size
        self.strides = pooling_stride
        self.thresholds = 0.9
        self.method = pooling_method

    @pytest.mark.xfail(reason="Not implemented.")
    def test_max_pooling(self, numpy_convolution_result, default_pooling_params):
        assert False

    @pytest.mark.xfail(reason="Not implemented.")
    def test_average_pooling(self, numpy_convolution_result, default_pooling_params):
        assert False

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
        scaffold.add_brick(BaseP_Input(np.array([self.pvector]),p=self.basep,bits=self.bits,collapse_binary=False,name="I",time_dimension=False),"input")
        scaffold.add_brick(convolution_2d(self.pshape,self.filters,self.conv_thresholds,self.basep,self.bits,name="convolution_",mode=self.mode),[(0, 0)],output=True)
        scaffold.add_brick(pooling_2d(self.pool_size,self.strides,thresholds=self.thresholds,name="pool_",method=self.method),[(1,0)],output=True)

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
    
    def get_stride_positions(self, pixel_dim):
        return np.arange(0, pixel_dim, self.strides, dtype=int)
    
    def get_output_size(self, pixel_dim):
        return int(np.floor(1.0 + (np.float64(pixel_dim) - self.pool_size)/self.strides))

    def get_output_shape(self):
        return (self.get_output_size(self.pixel_dim1), self.get_output_size(self.pixel_dim2))
    
    def get_expected_pooling_answer(self, pool_input):
        row_stride_positions = self.get_stride_positions(self.pixel_dim1)
        col_stride_positions = self.get_stride_positions(self.pixel_dim2)
        output_shape = self.get_output_shape()
        expected = []

        if self.method == "max":
            for row in row_stride_positions[:output_shape[0]]:
                for col in col_stride_positions[:output_shape[1]]:
                    expected.append(np.any(pool_input[row:row+self.pool_size,col:col+self.pool_size]))

            expected = np.reshape(expected, output_shape).astype(int)
            expected = (np.array(expected, dtype=int) > self.thresholds).astype(float)

        elif self.method == "average": #TODO : fix this code. 
            weights = np.ones((self.pool_size,self.pool_size),dtype=float) / float(self.pool_size*self.pool_size)
            for row in row_stride_positions[:output_shape[0]]:
                for col in col_stride_positions[:output_shape[1]]:
                    expected.append(np.dot(weights.flatten(),pool_input[row:row+self.pool_size,col:col+self.pool_size].astype(int).flatten()))

            expected = np.reshape(expected, output_shape).astype(float)
        else:
            print(f"'method' class member variable must be either 'max' or 'average'. But it is {self.method}.")
            raise ValueError("Unrecognized 'method' class member variable.")
            
        return np.array(expected)
    
    def get_expected_spikes(self,expected_ans):
        ans = np.array(expected_ans)
        spikes = ans[ ans > self.thresholds].astype(float)
        return list(2.0 * np.ones(spikes.shape))    

