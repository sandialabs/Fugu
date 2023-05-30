import numpy as np
import pytest
from scipy.signal import convolve2d

from fugu.backends import snn_Backend
from fugu.bricks.convolution_bricks import convolution_1d, convolution_2d
from fugu.bricks.input_bricks import BaseP_Input
from fugu.bricks.pooling_bricks import pooling_1d, pooling_2d
from fugu.scaffold import Scaffold


class Test_Pooling1D:

    @pytest.fixture
    def convolution_mode(self):
        return "full"
    
    @pytest.fixture
    def default_convolution_params(self, convolution_mode):
        self.basep = 3
        self.bits = 3
        self.filters = [2, 3]
        self.mode = convolution_mode
        self.pvector = [2, 3, 7, 4, 6, 2]

    @pytest.fixture(params=["fixed", "random"])
    def numpy_convolution_result(self, default_convolution_params, request):
        self.conv_ans = np.convolve(self.pvector, self.filters, mode=self.mode)
        self.pixel_size = len(self.conv_ans)
        self.conv_spike_pos = self.make_convolution_spike_positions_vector(request.param)
        self.conv_thresholds = self.conv_ans - self.conv_spike_pos
        return self.conv_ans > self.conv_thresholds        

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

    @pytest.mark.parametrize("pooling_size", np.arange(1,8))
    @pytest.mark.parametrize("pooling_stride", np.arange(1,8))
    def test_max_pooling(self, numpy_convolution_result, default_pooling_params):
        
        pool_input = numpy_convolution_result

        # Check calculations
        expected_ans = self.get_expected_pooling_answer(pool_input)
        expected_spikes = self.get_expected_spikes(expected_ans)

        # get output positions in result
        result = self.run_pooling_1d()
        output_positions = self.get_output_spike_positions()         
        output_mask = self.get_output_mask(output_positions, result)        

        calculated_spikes = list(result[output_mask].to_numpy()[:, 0])
        assert expected_spikes == calculated_spikes

    @pytest.mark.parametrize("pooling_size", np.arange(1,8))
    @pytest.mark.parametrize("pooling_stride", np.arange(1,8))
    def test_max_pooling_vector_thresholds(self, numpy_convolution_result, default_pooling_params):

        pool_input = numpy_convolution_result

        # Check calculations
        expected_ans = self.get_expected_pooling_answer(pool_input)
        expected_spikes = self.get_expected_spikes(expected_ans)   

        # pooling portion
        output_size = self.get_output_size()
        self.thresholds = 0.9 * np.ones((output_size,), dtype=float)
        result = self.run_pooling_1d()

        # get output positions in result
        output_positions = self.get_output_spike_positions()         
        output_mask = self.get_output_mask(output_positions, result)     

        calculated_spikes = list(result[output_mask].to_numpy()[:, 0])
        assert expected_spikes == calculated_spikes

    @pytest.mark.parametrize("pooling_method", ["average"])
    @pytest.mark.parametrize("pooling_size", np.arange(1,8))
    @pytest.mark.parametrize("pooling_stride", np.arange(1,8))
    def test_average_pooling(self, numpy_convolution_result, default_pooling_params):

        pool_input = numpy_convolution_result

        # Check calculations
        expected_ans = self.get_expected_pooling_answer(pool_input)        

        self.thresholds = self.make_random_thresholds_vector(expected_ans)
        result = self.run_pooling_1d()

        # get output positions in result
        output_positions = self.get_output_spike_positions()
        output_mask = self.get_output_mask(output_positions, result)       

        expected_spikes = self.get_expected_spikes(expected_ans) 
        calculated_spikes = list(result[output_mask].to_numpy()[:, 0])
        assert expected_spikes == calculated_spikes

    @pytest.mark.parametrize("pooling_size", np.arange(1,8))
    @pytest.mark.parametrize("pooling_stride", np.arange(1,8))
    @pytest.mark.parametrize("convolution_mode", ["full"])
    def test_max_pooling_with_full_convolution_mode(self, numpy_convolution_result, default_pooling_params):

        pool_input = numpy_convolution_result

        # Check calculations
        expected_ans = self.get_expected_pooling_answer(pool_input)
        expected_spikes = self.get_expected_spikes(expected_ans)  

        result = self.run_pooling_1d()

        # get output positions in result
        output_positions = self.get_output_spike_positions()
        output_mask = self.get_output_mask(output_positions, result)       

        calculated_spikes = list(result[output_mask].to_numpy()[:, 0])
        assert expected_spikes == calculated_spikes

    @pytest.mark.parametrize("pooling_size", np.arange(1,6))
    @pytest.mark.parametrize("pooling_stride", np.arange(1,6))
    @pytest.mark.parametrize("convolution_mode", ["valid"])
    def test_max_pooling_with_valid_convolution_mode(self, numpy_convolution_result, default_pooling_params):

        pool_input = numpy_convolution_result

        # Check calculations
        expected_ans = self.get_expected_pooling_answer(pool_input)
        expected_spikes = self.get_expected_spikes(expected_ans)  

        result = self.run_pooling_1d()

        # get output positions in result
        output_positions = self.get_output_spike_positions()
        output_mask = self.get_output_mask(output_positions, result)       

        calculated_spikes = list(result[output_mask].to_numpy()[:, 0])
        assert expected_spikes == calculated_spikes

    @pytest.mark.parametrize("pooling_size", np.arange(1,7))
    @pytest.mark.parametrize("pooling_stride", np.arange(1,7))
    @pytest.mark.parametrize("convolution_mode", ["same"])
    def test_max_pooling_with_same_convolution_mode(self, numpy_convolution_result, default_pooling_params):

        pool_input = numpy_convolution_result

        # Check calculations
        expected_ans = self.get_expected_pooling_answer(pool_input)
        expected_spikes = self.get_expected_spikes(expected_ans)  

        result = self.run_pooling_1d()

        # get output positions in result
        output_positions = self.get_output_spike_positions()
        output_mask = self.get_output_mask(output_positions, result)       

        calculated_spikes = list(result[output_mask].to_numpy()[:, 0])
        assert expected_spikes == calculated_spikes

    @pytest.mark.parametrize("pooling_method", ["average"])
    @pytest.mark.parametrize("pooling_size", np.arange(1,8))
    @pytest.mark.parametrize("pooling_stride", np.arange(1,8))
    @pytest.mark.parametrize("convolution_mode", ["full"])
    def test_average_pooling_with_full_convolution_mode(self, numpy_convolution_result, default_pooling_params):

        pool_input = numpy_convolution_result

        # Check calculations
        expected_ans = self.get_expected_pooling_answer(pool_input)       

        # begin pooling brick calculation   
        self.thresholds = self.make_random_thresholds_vector(expected_ans)
        result = self.run_pooling_1d()

        # get output positions in result
        output_positions = self.get_output_spike_positions()
        output_mask = self.get_output_mask(output_positions, result)       

        expected_spikes = self.get_expected_spikes(expected_ans)        
        calculated_spikes = list(result[output_mask].to_numpy()[:, 0])
        assert expected_spikes == calculated_spikes

    @pytest.mark.parametrize("pooling_method", ["average"])
    @pytest.mark.parametrize("pooling_size", np.arange(1,6))
    @pytest.mark.parametrize("pooling_stride", np.arange(1,6))
    @pytest.mark.parametrize("convolution_mode", ["valid"])
    def test_average_pooling_with_valid_convolution_mode(self, numpy_convolution_result, default_pooling_params):

        pool_input = numpy_convolution_result

        # Check calculations
        expected_ans = self.get_expected_pooling_answer(pool_input)       

        # begin pooling brick calculation   
        self.thresholds = self.make_random_thresholds_vector(expected_ans)
        result = self.run_pooling_1d()

        # get output positions in result
        output_positions = self.get_output_spike_positions()
        output_mask = self.get_output_mask(output_positions, result)       

        expected_spikes = self.get_expected_spikes(expected_ans)        
        calculated_spikes = list(result[output_mask].to_numpy()[:, 0])
        assert expected_spikes == calculated_spikes

    @pytest.mark.parametrize("pooling_method", ["average"])
    @pytest.mark.parametrize("pooling_size", np.arange(1,7))
    @pytest.mark.parametrize("pooling_stride", np.arange(1,7))
    @pytest.mark.parametrize("convolution_mode", ["same"])
    def test_average_pooling_with_same_convolution_mode(self, numpy_convolution_result, default_pooling_params):

        pool_input = numpy_convolution_result

        # Check calculations
        expected_ans = self.get_expected_pooling_answer(pool_input)       

        # begin pooling brick calculation   
        self.thresholds = self.make_random_thresholds_vector(expected_ans)
        result = self.run_pooling_1d()

        # get output positions in result
        output_positions = self.get_output_spike_positions()
        output_mask = self.get_output_mask(output_positions, result)       

        expected_spikes = self.get_expected_spikes(expected_ans)        
        calculated_spikes = list(result[output_mask].to_numpy()[:, 0])
        assert expected_spikes == calculated_spikes

    @pytest.mark.parametrize("length", [1,2,4,5])
    def test_thresholds_shape(self, default_convolution_params, default_pooling_params, length):
        '''
        Correct pooling threshold shape is (3,)
        '''
        self.conv_thresholds = np.array([0.,1.,0.,1.,0.,0.,0.])
        with pytest.raises(ValueError):
            self.thresholds = 0.9*np.ones((length,))
            self.run_pooling_1d()

    @pytest.mark.parametrize("thresholds", [1.1,2.1,np.ones((3,),dtype=float)])
    def test_max_pooling_thresholds_changes(self, default_convolution_params, default_pooling_params, thresholds):
        '''
        Max pooling threshold must be < 0.9 for the "or" operation to be performed correctly. Otherwise, the brick no longer behaves as "max" operation.
        '''
        self.conv_thresholds = np.array([0.,1.,0.,1.,0.,0.,0.])
        with pytest.raises(ValueError):
            self.thresholds = thresholds
            self.run_pooling_1d()

    def test_simple_explicit_max_pooling(self):
        '''
        Explicit invocation of the pooling brick unit test. Removes the complexity of the tests above to show the 
        basic structure of general unit tests for the pooling brick.
        '''
        self.basep = 3
        self.bits = 3
        self.filters = [2, 3]
        self.mode = "full"
        self.pvector = [2, 3, 7, 4, 6, 2]

        self.pool_size = 2
        self.strides = 2
        self.thresholds = 0.9
        self.method = "max"

        self.conv_ans = np.convolve(self.pvector, self.filters, mode=self.mode)
        self.pixel_size = len(self.conv_ans)
        self.conv_spike_pos = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0])

        self.conv_thresholds = self.conv_ans - self.conv_spike_pos
        pool_input = self.conv_ans > self.conv_thresholds
        
        # Check calculations
        expected_ans = self.get_expected_pooling_answer(pool_input)
        expected_spikes = self.get_expected_spikes(expected_ans) 

        result = self.run_pooling_1d()

        # get output positions in result
        output_positions = self.get_output_spike_positions()         
        output_mask = self.get_output_mask(output_positions, result)       

        calculated_spikes = list(result[output_mask].to_numpy()[:, 0])
        assert expected_spikes == calculated_spikes

    def test_simple_explicit_average_pooling(self):
        '''
        Explicit invocation of the pooling brick unit test. Removes the complexity of the tests above to show the 
        basic structure of general unit tests for the pooling brick.
        '''
        self.basep = 3
        self.bits = 3
        self.filters = [2, 3]
        self.mode = "full"
        self.pvector = [2, 3, 7, 4, 6, 2]

        self.pool_size = 2
        self.strides = 2
        self.thresholds = np.array([0.4,0.9,0.9])
        self.method = "average"

        self.conv_ans = np.convolve(self.pvector, self.filters, mode=self.mode)
        self.pixel_size = len(self.conv_ans)
        self.conv_spike_pos = np.array([0.0, 0.1, 0.0, 0.1, 0.0, 0.0, 0.0])

        self.conv_thresholds = self.conv_ans - self.conv_spike_pos
        pool_input = self.conv_ans > self.conv_thresholds
        
        # Check calculations
        expected_ans = self.get_expected_pooling_answer(pool_input)
        expected_spikes = self.get_expected_spikes(expected_ans) 

        result = self.run_pooling_1d()

        # get output positions in result
        output_positions = self.get_output_spike_positions()         
        output_mask = self.get_output_mask(output_positions, result)

        calculated_spikes = list(result[output_mask].to_numpy()[:, 0])
        assert expected_spikes == calculated_spikes

    def make_random_thresholds_vector(self, initial_vector):
        output_size = self.get_output_size()
        thresholds = initial_vector.copy()
        random_ids = np.random.choice(output_size, np.ceil(0.5*output_size).astype(int), replace=False)
        thresholds[random_ids] = thresholds[random_ids] - 0.1
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
            conv_spike_pos[[1, 3]] = 0.1
        elif case.lower() == "random":
            random_ids = np.random.choice(conv_spike_pos.size, np.ceil(0.5*conv_spike_pos.size).astype(int), replace=False)
            conv_spike_pos[random_ids] = 0.1
        else:
            print(f"Case parameter is either 'fixed' or 'random'. But you provided {case}.")
            raise ValueError("Unrecognized 'case' parameter value provided.")
        
        return conv_spike_pos    
    
    def get_output_neuron_numbers(self):
        neuron_numbers = []
        for key in self.graph.nodes.keys():
            if key.startswith('pool_p'):
                neuron_numbers.append(self.graph.nodes[key]['neuron_number'])

        return np.array(neuron_numbers)

    def get_output_spike_positions(self):
        neuron_numbers = self.get_output_neuron_numbers()
        output_ini_position = np.amin(neuron_numbers)
        output_end_position = np.amax(neuron_numbers)
        return [output_ini_position,output_end_position]

    def get_output_mask(self,output_positions, result):
        ini = output_positions[0]
        end = output_positions[1]
        return (result["neuron_number"] >= ini) & (result["neuron_number"] <= end)

    def run_pooling_1d(self):
        scaffold = Scaffold()
        scaffold.add_brick(BaseP_Input(np.array([self.pvector]),p=self.basep,bits=self.bits,collapse_binary=False,name="Input0",time_dimension=False),"input")
        scaffold.add_brick(convolution_1d(self.pvector,self.filters,self.conv_thresholds,self.basep,self.bits,name="convolution_",mode=self.mode),[(0, 0)],output=True)
        scaffold.add_brick(pooling_1d(self.pool_size,self.strides,thresholds=self.thresholds,name="pool_",method=self.method),[(1,0)],output=True)

        self.graph = scaffold.lay_bricks()
        scaffold.summary(verbose=1)
        backend = snn_Backend()
        backend_args = {}
        backend.compile(scaffold, backend_args)
        result = backend.run(5)
        return result

    def get_stride_positions(self):
        return np.arange(0, self.pixel_size, self.strides, dtype=int)
    
    def get_output_size(self):
        return int(np.floor(1.0 + (np.float64(self.pixel_size) - self.pool_size)/self.strides))
        
    def get_expected_pooling_answer(self, pool_input):
        stride_positions = self.get_stride_positions()
        output_size = self.get_output_size()
        expected = []
        
        if self.method == "max":
            for pos in stride_positions[:output_size]:
                expected.append(np.any(pool_input[pos:pos+self.pool_size]))

            expected = (np.array(expected, dtype=int) > self.thresholds).astype(int)

        elif self.method == "average":
            weights = np.ones((self.pool_size,),dtype=float) / float(self.pool_size)
            for pos in stride_positions[:output_size]:
                expected.append(np.dot(weights,pool_input[pos:pos+self.pool_size].astype(int)))

        else:
            print(f"'method' class member variable must be either 'max' or 'average'. But it is {self.method}.")
            raise ValueError("Unrecognized 'method' class member variable.")
                
        return np.array(expected)
    
    def get_expected_spikes(self,expected_ans):
        ans = np.array(expected_ans)
        spikes = ans[ ans > self.thresholds].astype(float)
        return list(2.0 * np.ones(spikes.shape))

class Test_Pooling2D:

    @pytest.fixture
    def convolution_mode(self):
        return "full"

    @pytest.fixture
    def default_convolution_params(self, convolution_mode):
        self.basep = 3
        self.bits = 3
        self.filters = [[2, 3],[4,5]]
        self.mode = convolution_mode
        self.pvector = [[1,1,4,6],[6,2,4,2],[2,3,5,4],[6,1,6,3]]

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

    @pytest.mark.parametrize("pooling_size", np.arange(1,6))
    @pytest.mark.parametrize("pooling_stride", np.arange(1,6))
    def test_max_pooling(self, numpy_convolution_result, default_pooling_params):
        
        pool_input = numpy_convolution_result

        # Check calculations
        expected_ans = self.get_expected_pooling_answer(pool_input)
        expected_spikes = self.get_expected_spikes(expected_ans)

        # get output positions in result
        result = self.run_pooling_2d()
        output_positions = self.get_output_spike_positions()         
        output_mask = self.get_output_mask(output_positions, result)        

        calculated_spikes = list(result[output_mask].to_numpy()[:, 0])
        assert expected_spikes == calculated_spikes

    @pytest.mark.parametrize("pooling_method", ["average"])
    @pytest.mark.parametrize("pooling_size", np.arange(1,6))
    @pytest.mark.parametrize("pooling_stride", np.arange(1,6))
    def test_average_pooling(self, numpy_convolution_result, default_pooling_params):

        pool_input = numpy_convolution_result

        # Check calculations
        expected_ans = self.get_expected_pooling_answer(pool_input)        

        self.thresholds = self.make_random_thresholds_vector(expected_ans)
        result = self.run_pooling_2d()

        # get output positions in result
        output_positions = self.get_output_spike_positions()
        output_mask = self.get_output_mask(output_positions, result)       

        expected_spikes = self.get_expected_spikes(expected_ans) 
        calculated_spikes = list(result[output_mask].to_numpy()[:, 0])
        assert expected_spikes == calculated_spikes

    @pytest.mark.parametrize("pooling_size", np.arange(1,6))
    @pytest.mark.parametrize("pooling_stride", np.arange(1,6))
    def test_max_pooling_vector_thresholds(self, numpy_convolution_result, default_pooling_params):

        pool_input = numpy_convolution_result

        # Check calculations
        expected_ans = self.get_expected_pooling_answer(pool_input)
        expected_spikes = self.get_expected_spikes(expected_ans)   

        # pooling portion
        self.thresholds = 0.9 * np.ones(self.get_output_shape(), dtype=float)
        result = self.run_pooling_2d()

        # get output positions in result
        output_positions = self.get_output_spike_positions()         
        output_mask = self.get_output_mask(output_positions, result)     

        calculated_spikes = list(result[output_mask].to_numpy()[:, 0])
        assert expected_spikes == calculated_spikes

    @pytest.mark.parametrize("pooling_method", ["max"])
    @pytest.mark.parametrize("pooling_size", np.arange(1,6))
    @pytest.mark.parametrize("pooling_stride", np.arange(1,6))
    @pytest.mark.parametrize("convolution_mode", ["full"])
    @pytest.mark.parametrize("spike_positions_vector", ["random"])
    def test_max_pooling_with_full_convolution_mode(self, numpy_convolution_result, default_pooling_params):
        
        pool_input = numpy_convolution_result

        # Check calculations
        expected_ans = self.get_expected_pooling_answer(pool_input)
        expected_spikes = self.get_expected_spikes(expected_ans)

        # get output positions in result
        result = self.run_pooling_2d()
        output_positions = self.get_output_spike_positions()         
        output_mask = self.get_output_mask(output_positions, result)        

        calculated_spikes = list(result[output_mask].to_numpy()[:, 0])
        assert expected_spikes == calculated_spikes

    @pytest.mark.parametrize("pooling_method", ["max"])
    @pytest.mark.parametrize("pooling_size", np.arange(1,4))
    @pytest.mark.parametrize("pooling_stride", np.arange(1,4))
    @pytest.mark.parametrize("convolution_mode", ["valid"])
    @pytest.mark.parametrize("spike_positions_vector", ["random"])
    def test_max_pooling_with_valid_convolution_mode(self, numpy_convolution_result, default_pooling_params):
        
        pool_input = numpy_convolution_result

        # Check calculations
        expected_ans = self.get_expected_pooling_answer(pool_input)
        expected_spikes = self.get_expected_spikes(expected_ans)

        # get output positions in result
        result = self.run_pooling_2d()
        output_positions = self.get_output_spike_positions()         
        output_mask = self.get_output_mask(output_positions, result)        

        calculated_spikes = list(result[output_mask].to_numpy()[:, 0])
        assert expected_spikes == calculated_spikes

    @pytest.mark.parametrize("pooling_method", ["max"])
    @pytest.mark.parametrize("pooling_size", np.arange(1,5))
    @pytest.mark.parametrize("pooling_stride", np.arange(1,5))
    @pytest.mark.parametrize("convolution_mode", ["same"])
    @pytest.mark.parametrize("spike_positions_vector", ["random"])
    def test_max_pooling_with_same_convolution_mode(self, numpy_convolution_result, default_pooling_params):
        
        pool_input = numpy_convolution_result

        # Check calculations
        expected_ans = self.get_expected_pooling_answer(pool_input)
        expected_spikes = self.get_expected_spikes(expected_ans)

        # get output positions in result
        result = self.run_pooling_2d()
        output_positions = self.get_output_spike_positions()         
        output_mask = self.get_output_mask(output_positions, result)        

        calculated_spikes = list(result[output_mask].to_numpy()[:, 0])
        assert expected_spikes == calculated_spikes

    @pytest.mark.parametrize("pooling_method", ["average"])
    @pytest.mark.parametrize("pooling_size", np.arange(1,6))
    @pytest.mark.parametrize("pooling_stride", np.arange(1,6))
    @pytest.mark.parametrize("convolution_mode", ["full"])
    @pytest.mark.parametrize("spike_positions_vector", ["random"])
    def test_average_pooling_with_full_convolution_mode(self, numpy_convolution_result, default_pooling_params):

        pool_input = numpy_convolution_result

        # Check calculations
        expected_ans = self.get_expected_pooling_answer(pool_input)        

        self.thresholds = self.make_random_thresholds_vector(expected_ans)
        result = self.run_pooling_2d()

        # get output positions in result
        output_positions = self.get_output_spike_positions()
        output_mask = self.get_output_mask(output_positions, result)       

        expected_spikes = self.get_expected_spikes(expected_ans) 
        calculated_spikes = list(result[output_mask].to_numpy()[:, 0])
        assert expected_spikes == calculated_spikes

    @pytest.mark.parametrize("pooling_method", ["average"])
    @pytest.mark.parametrize("pooling_size", np.arange(1,4))
    @pytest.mark.parametrize("pooling_stride", np.arange(1,4))
    @pytest.mark.parametrize("convolution_mode", ["valid"])
    @pytest.mark.parametrize("spike_positions_vector", ["random"])
    def test_average_pooling_with_valid_convolution_mode(self, numpy_convolution_result, default_pooling_params):

        pool_input = numpy_convolution_result

        # Check calculations
        expected_ans = self.get_expected_pooling_answer(pool_input)        

        self.thresholds = self.make_random_thresholds_vector(expected_ans)
        result = self.run_pooling_2d()

        # get output positions in result
        output_positions = self.get_output_spike_positions()
        output_mask = self.get_output_mask(output_positions, result)       

        expected_spikes = self.get_expected_spikes(expected_ans) 
        calculated_spikes = list(result[output_mask].to_numpy()[:, 0])
        assert expected_spikes == calculated_spikes

    @pytest.mark.parametrize("pooling_method", ["average"])
    @pytest.mark.parametrize("pooling_size", np.arange(1,5))
    @pytest.mark.parametrize("pooling_stride", np.arange(1,5))
    @pytest.mark.parametrize("convolution_mode", ["same"])
    @pytest.mark.parametrize("spike_positions_vector", ["random"])
    def test_average_pooling_with_same_convolution_mode(self, numpy_convolution_result, default_pooling_params):

        pool_input = numpy_convolution_result

        # Check calculations
        expected_ans = self.get_expected_pooling_answer(pool_input)        

        self.thresholds = self.make_random_thresholds_vector(expected_ans)
        result = self.run_pooling_2d()

        # get output positions in result
        output_positions = self.get_output_spike_positions()
        output_mask = self.get_output_mask(output_positions, result)       

        expected_spikes = self.get_expected_spikes(expected_ans) 
        calculated_spikes = list(result[output_mask].to_numpy()[:, 0])
        assert expected_spikes == calculated_spikes

    @pytest.mark.parametrize("convolution_mode", ["full"])
    @pytest.mark.parametrize("shape", [(1,), (2,), (3,3), (4,4), (5,5)])
    def test_thresholds_shape(self, default_convolution_params, default_pooling_params, shape):
        '''
        Correct pooling threshold shape is (2,2)
        '''
        self.conv_thresholds = np.array([[0.9,1.,1.,1.,0.9],[1.,1.,0.9,0.9,0.9],[0.9,0.9,0.9,1.,0.9],[1.,0.9,1.,1.,0.9],[0.9,0.9,1.,1.,1.]],dtype=float)
        with pytest.raises(ValueError):
            self.thresholds = 0.9*np.ones(shape)
            self.run_pooling_2d()

    @pytest.mark.parametrize("convolution_mode", ["full"])
    @pytest.mark.parametrize("thresholds", [1.1,2.1,np.ones((2,2),dtype=float)])
    def test_max_pooling_thresholds_changes(self, default_convolution_params, default_pooling_params, thresholds):
        '''
        Max pooling threshold must be < 0.9 for the "or" operation to be performed correctly. Otherwise, the brick no longer behaves as "max" operation.
        '''
        self.conv_thresholds = np.array([[0.9,1.,1.,1.,0.9],[1.,1.,0.9,0.9,0.9],[0.9,0.9,0.9,1.,0.9],[1.,0.9,1.,1.,0.9],[0.9,0.9,1.,1.,1.]],dtype=float)
        with pytest.raises(ValueError):
            self.thresholds = thresholds
            self.run_pooling_2d()

    def test_simple_explicit_max_pooling(self):
        '''
        Explicit invocation of the pooling brick unit test. Removes the complexity of the tests above to show the 
        basic structure of general unit tests for the pooling brick.
        '''
        self.basep = 3
        self.bits = 3
        self.filters = [[2, 3],[4,5]]
        self.mode = "full"
        self.pvector = [[1, 1, 4, 6],[6,2,4,2],[2,3,5,4],[6,1,6,3]]

        self.pool_size = 2
        self.strides = 2
        self.thresholds = 0.9
        self.method = "max"

        self.conv_ans = convolve2d(self.pvector, self.filters, mode=self.mode)
        self.pixel_shape = self.conv_ans.shape
        self.pixel_dim1 = self.pixel_shape[0]
        self.pixel_dim2 = self.pixel_shape[1]
        self.conv_spike_pos = 0.1*np.array([[1,0,0,0,1],[0,0,1,1,1],[1,1,1,0,1],[0,1,0,0,1],[1,1,0,0,0]],dtype=float)

        self.conv_thresholds = self.conv_ans - self.conv_spike_pos
        pool_input = self.conv_ans > self.conv_thresholds
        
        # Check calculations
        # expected_ans = self.get_expected_pooling_answer(pool_input)
        expected_ans = np.array([[1, 1],[1, 1]], dtype=float)
        expected_spikes = self.get_expected_spikes(expected_ans)

        result = self.run_pooling_2d()

        # get output positions in result
        output_positions = self.get_output_spike_positions()         
        output_mask = self.get_output_mask(output_positions, result)       

        calculated_spikes = list(result[output_mask].to_numpy()[:, 0])
        assert expected_spikes == calculated_spikes

    def test_simple_explicit_average_pooling(self):
        '''
        Explicit invocation of the pooling brick unit test. Removes the complexity of the tests above to show the 
        basic structure of general unit tests for the pooling brick.
        '''
        self.basep = 3
        self.bits = 3
        self.filters = [[2, 3],[4,5]]
        self.mode = "full"
        self.pvector = [[1, 1, 4, 6],[6,2,4,2],[2,3,5,4],[6,1,6,3]]

        self.pool_size = 2
        self.strides = 2
        self.thresholds = np.array([[0.20,0.9],[0.7,0.30]])      
        self.method = "average"

        self.conv_ans = convolve2d(self.pvector, self.filters, mode=self.mode)
        self.pixel_shape = self.conv_ans.shape
        self.pixel_dim1 = self.pixel_shape[0]
        self.pixel_dim2 = self.pixel_shape[1]
        self.conv_spike_pos = 0.1*np.array([[1,0,0,0,1],[0,0,1,1,1],[1,1,1,0,1],[0,1,0,0,1],[1,1,0,0,0]],dtype=float)

        self.conv_thresholds = self.conv_ans - self.conv_spike_pos
        pool_input = self.conv_ans > self.conv_thresholds
        
        # Check calculations
        # expected_ans = self.get_expected_pooling_answer(pool_input)
        expected_ans = np.array([[1, 0],[1, 0]], dtype=float)
        expected_spikes = self.get_expected_spikes(expected_ans)

        result = self.run_pooling_2d()

        # get output positions in result
        output_positions = self.get_output_spike_positions()         
        output_mask = self.get_output_mask(output_positions, result)       

        calculated_spikes = list(result[output_mask].to_numpy()[:, 0])
        assert expected_spikes == calculated_spikes

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
        scaffold.add_brick(BaseP_Input(np.array([self.pvector]),p=self.basep,bits=self.bits,collapse_binary=False,name="Input0",time_dimension=False),"input")
        scaffold.add_brick(convolution_2d(self.pvector,self.filters,self.conv_thresholds,self.basep,self.bits,name="convolution_",mode=self.mode),[(0, 0)],output=True)
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

