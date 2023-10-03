import numpy as np
import pytest

from fugu.backends import snn_Backend
from fugu.bricks.convolution_bricks import convolution_1d, convolution_2d
from fugu.bricks.dense_bricks import dense_layer_1d, dense_layer_2d
from fugu.bricks.input_bricks import BaseP_Input
from fugu.bricks.pooling_bricks import pooling_1d, pooling_2d
from fugu.scaffold import Scaffold

convolve2d = pytest.importorskip("scipy.signal", reason=f"Scipy package not installed. Skipping test file {__file__} because of module dependency.").convolve2d

class Test_DenseLayer1D:

    def test_simple_explicit_dense_layer_example_1(self):
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
        self.pool_strides = 2
        self.pool_thresholds = 0.9
        self.pool_method = "max"

        self.dense_thresholds = [0.9, 0.9, 0.9]
        self.dense_weights = [[1.0, 1.0, 1.0],[1.0, 1.0, 1.0],[1.0, 1.0, 1.0]]

        # self.dense_thresholds = [2.9, 8.9, 15.0]
        # self.dense_weights = [[1.0, 2.0, 3.0],[4.0, 5.0, 6.0],[7.0, 8.0, 9.0]]

        self.conv_ans = np.convolve(self.pvector, self.filters, mode=self.mode)
        self.pixel_size = len(self.conv_ans)
        self.conv_spike_pos = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0])

        self.conv_thresholds = self.conv_ans - self.conv_spike_pos
        pool_input = self.conv_ans > self.conv_thresholds
        
        # Check calculations
        dense_input = self.get_expected_pooling_answer(pool_input)
        # expected_spikes = self.get_expected_spikes(expected_ans) 

        dense_thresholds = np.array(self.dense_thresholds)
        dense_weights = np.array(self.dense_weights)
        dense_ans = np.matmul(dense_weights, dense_input)
        # dense_ans = (dense_weights * dense_input > dense_thresholds).astype(int)
        expected_spikes = self.get_expected_spikes(dense_ans, dense_thresholds)

        result = self.run_dense_layer_1d(dense_input.shape)

        # get output positions in result
        output_positions = self.get_output_spike_positions()         
        output_mask = self.get_output_mask(output_positions, result)       

        calculated_spikes = list(result[output_mask].to_numpy()[:, 0])
        assert expected_spikes == calculated_spikes

    def test_simple_explicit_dense_layer_example_2(self):
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
        self.pool_strides = 2
        self.pool_thresholds = 0.9
        self.pool_method = "max"

        self.dense_thresholds = [2.9, 8.9, 15.0]
        self.dense_weights = [[1.0, 2.0, 3.0],[4.0, 5.0, 6.0],[7.0, 8.0, 9.0]]

        self.conv_ans = np.convolve(self.pvector, self.filters, mode=self.mode)
        self.pixel_size = len(self.conv_ans)
        self.conv_spike_pos = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0])

        self.conv_thresholds = self.conv_ans - self.conv_spike_pos
        pool_input = self.conv_ans > self.conv_thresholds
        
        # Check calculations
        dense_input = self.get_expected_pooling_answer(pool_input)

        dense_thresholds = np.array(self.dense_thresholds)
        dense_weights = np.array(self.dense_weights)
        dense_ans = np.matmul(dense_weights, dense_input)
        expected_spikes = self.get_expected_spikes(dense_ans, dense_thresholds)

        result = self.run_dense_layer_1d(dense_input.shape)

        # get output positions in result
        output_positions = self.get_output_spike_positions()         
        output_mask = self.get_output_mask(output_positions, result)       

        calculated_spikes = list(result[output_mask].to_numpy()[:, 0])
        assert expected_spikes == calculated_spikes

    #TODO: Unit test that checks dense thresholds shape
    #TODO: Unit test that checks dense weights shape
    #TODO: Unit test that test dense brick with a scalar threshold value
    #TODO: Unit test that test dense brick with a scalar weight value
    #TODO: Unit test that uses different input/output neuron shapes

    def run_dense_layer_1d(self, output_shape):
        scaffold = Scaffold()
        scaffold.add_brick(BaseP_Input(np.array([self.pvector]),p=self.basep,bits=self.bits,collapse_binary=False,name="Input0",time_dimension=False),"input")
        scaffold.add_brick(convolution_1d(self.pvector,self.filters,self.conv_thresholds,self.basep,self.bits,name="convolution_",mode=self.mode),[(0, 0)],output=True)
        scaffold.add_brick(pooling_1d(self.pool_size,self.pool_strides,thresholds=self.pool_thresholds,name="pool_",method=self.pool_method),[(1,0)],output=True)
        scaffold.add_brick(dense_layer_1d(output_shape,weights=self.dense_weights,thresholds=self.dense_thresholds,name="dense_"),[(2,0)],output=True)

        self.graph = scaffold.lay_bricks()
        scaffold.summary(verbose=1)
        backend = snn_Backend()
        backend_args = {}
        backend.compile(scaffold, backend_args)
        result = backend.run(5)
        return result

    def get_output_neuron_numbers(self):
        neuron_numbers = []
        for key in self.graph.nodes.keys():
            if key.startswith('dense_d'):
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

    def get_stride_positions(self):
        return np.arange(0, self.pixel_size, self.pool_strides, dtype=int)
    
    def get_output_size(self):
        return int(np.floor(1.0 + (np.float64(self.pixel_size) - self.pool_size)/self.pool_strides))
    
    def get_expected_pooling_answer(self, pool_input):
        stride_positions = self.get_stride_positions()
        output_size = self.get_output_size()
        expected = []
        
        if self.pool_method == "max":
            for pos in stride_positions[:output_size]:
                expected.append(np.any(pool_input[pos:pos+self.pool_size]))

            expected = (np.array(expected, dtype=int) > self.pool_thresholds).astype(int)

        elif self.pool_method == "average":
            weights = np.ones((self.pool_size,),dtype=float) / float(self.pool_size)
            for pos in stride_positions[:output_size]:
                expected.append(np.dot(weights,pool_input[pos:pos+self.pool_size].astype(int)))

        else:
            print(f"'pool_method' class member variable must be either 'max' or 'average'. But it is {self.pool_method}.")
            raise ValueError("Unrecognized 'pool_method' class member variable.")
                
        return np.array(expected)
    
    def get_expected_spikes(self, expected_ans, thresholds):   
        ans = np.array(expected_ans)
        spikes = ans[ ans > thresholds].astype(float)
        return list(3.0 * np.ones(spikes.shape))
    
class Test_DenseLayer2D:
    def test_simple_explicit_dense_layer_example_1(self):
        '''
        Explicit invocation of the pooling brick unit test. Removes the complexity of the tests above to show the 
        basic structure of general unit tests for the pooling brick.
        '''
        self.basep = 3
        self.bits = 3
        self.filters = [[2, 3],[4,5]]
        self.mode = "full"
        self.pvector = [[1,1,4,6],[6,2,4,2],[2,3,5,4],[6,1,6,3]]

        self.pool_size = 2
        self.pool_strides = 2
        self.pool_thresholds = 0.9
        self.pool_method = "max"

        self.dense_thresholds = np.array([[1.9, 1.9],[1.9, 2.0]])
        self.dense_weights = np.ones((self.dense_thresholds.size,self.dense_thresholds.size), dtype=float)

        self.conv_ans = convolve2d(self.pvector, self.filters, mode=self.mode)
        self.pixel_shape = self.conv_ans.shape
        self.pixel_dim1 = self.pixel_shape[0]
        self.pixel_dim2 = self.pixel_shape[1]
        self.conv_spike_pos = np.zeros(self.conv_ans.shape)
        self.conv_spike_pos[[0,1,1,2,3,4],[1,0,1,2,3,4]] = 0.1

        self.conv_thresholds = self.conv_ans - self.conv_spike_pos
        pool_input = self.conv_ans > self.conv_thresholds
        
        # Check calculations
        dense_input = self.get_expected_pooling_answer(pool_input)

        dense_thresholds = np.array(self.dense_thresholds)
        dense_weights = np.array(self.dense_weights)
        dense_ans = np.matmul(dense_weights, dense_input.flatten()).reshape(dense_thresholds.shape)
        expected_spikes = self.get_expected_spikes(dense_ans, dense_thresholds)

        result = self.run_dense_layer_2d(dense_input.shape)

        # get output positions in result
        output_positions = self.get_output_spike_positions()         
        output_mask = self.get_output_mask(output_positions, result)       

        calculated_spikes = list(result[output_mask].to_numpy()[:, 0])
        assert expected_spikes == calculated_spikes

    def test_simple_explicit_dense_layer_example_2(self):
        '''
        Explicit invocation of the pooling brick unit test. Removes the complexity of the tests above to show the 
        basic structure of general unit tests for the pooling brick.
        '''
        self.basep = 3
        self.bits = 3
        self.filters = [[2, 3],[4,5]]
        self.mode = "full"
        self.pvector = [[1,1,4,6],[6,2,4,2],[2,3,5,4],[6,1,6,3]]

        self.pool_size = 2
        self.pool_strides = 2
        self.pool_thresholds = 0.9
        self.pool_method = "max"

        self.dense_thresholds = [[6.9, 18.9],[30.9, 43.0]]
        self.dense_weights = np.arange(1,17,dtype=float).reshape((4,4))

        self.conv_ans = convolve2d(self.pvector, self.filters, mode=self.mode)
        self.pixel_shape = self.conv_ans.shape
        self.pixel_dim1 = self.pixel_shape[0]
        self.pixel_dim2 = self.pixel_shape[1]
        self.conv_spike_pos = np.zeros(self.conv_ans.shape)
        self.conv_spike_pos[[0,0,1,1,2,3,4],[1,3,0,1,2,3,4]] = 0.1

        self.conv_thresholds = self.conv_ans - self.conv_spike_pos
        pool_input = self.conv_ans > self.conv_thresholds
        
        # Check calculations
        dense_input = self.get_expected_pooling_answer(pool_input)

        dense_thresholds = np.array(self.dense_thresholds)
        dense_weights = np.array(self.dense_weights)
        dense_ans = np.matmul(dense_weights, dense_input.flatten()).reshape(dense_thresholds.shape)
        expected_spikes = self.get_expected_spikes(dense_ans, dense_thresholds)

        result = self.run_dense_layer_2d(dense_input.shape)

        # get output positions in result
        output_positions = self.get_output_spike_positions()         
        output_mask = self.get_output_mask(output_positions, result)       

        calculated_spikes = list(result[output_mask].to_numpy()[:, 0])
        assert expected_spikes == calculated_spikes

    def test_dense_layer_thresholds_shape(self):
        self.basep = 3
        self.bits = 3
        self.filters = [[2, 3],[4,5]]
        self.mode = "full"
        self.pvector = [[1,1,4,6],[6,2,4,2],[2,3,5,4],[6,1,6,3]]

        self.pool_size = 2
        self.pool_strides = 2
        self.pool_thresholds = 0.9
        self.pool_method = "max"

        self.dense_thresholds = [[6.9],[30.9]]
        self.dense_weights = np.arange(1,17,dtype=float).reshape((4,4))

        self.conv_ans = convolve2d(self.pvector, self.filters, mode=self.mode)
        self.pixel_shape = self.conv_ans.shape
        self.pixel_dim1 = self.pixel_shape[0]
        self.pixel_dim2 = self.pixel_shape[1]
        self.conv_spike_pos = np.zeros(self.conv_ans.shape)
        self.conv_spike_pos[[0,0,1,1,2,3,4],[1,3,0,1,2,3,4]] = 0.1

        self.conv_thresholds = self.conv_ans - self.conv_spike_pos
        pool_input = self.conv_ans > self.conv_thresholds
        
        # Check calculations
        dense_input = self.get_expected_pooling_answer(pool_input)

        # dense_thresholds = np.array(self.dense_thresholds)
        # dense_weights = np.array(self.dense_weights)
        # dense_ans = np.matmul(dense_weights, dense_input.flatten()).reshape(dense_thresholds.shape)
        # expected_spikes = self.get_expected_spikes(dense_ans, dense_thresholds)

        with pytest.raises(ValueError):
            self.run_dense_layer_2d(dense_input.shape)

    @pytest.mark.parametrize("output_shape", [(2,2), (4,)])
    def test_dense_layer_scalar_thresholds(self, output_shape):
        self.basep = 3
        self.bits = 3
        self.filters = [[2, 3],[4,5]]
        self.mode = "full"
        self.pvector = [[1,1,4,6],[6,2,4,2],[2,3,5,4],[6,1,6,3]]

        self.pool_size = 2
        self.pool_strides = 2
        self.pool_thresholds = 0.9
        self.pool_method = "max"

        self.dense_thresholds = 42.9
        self.dense_weights = np.arange(1,17,dtype=float).reshape((4,4))

        self.conv_ans = convolve2d(self.pvector, self.filters, mode=self.mode)
        self.pixel_shape = self.conv_ans.shape
        self.pixel_dim1 = self.pixel_shape[0]
        self.pixel_dim2 = self.pixel_shape[1]
        self.conv_spike_pos = np.zeros(self.conv_ans.shape)
        self.conv_spike_pos[[0,0,1,1,2,3,4],[1,3,0,1,2,3,4]] = 0.1

        self.conv_thresholds = self.conv_ans - self.conv_spike_pos
        pool_input = self.conv_ans > self.conv_thresholds
        
        # Check calculations
        dense_input = self.get_expected_pooling_answer(pool_input)

        dense_thresholds = self.dense_thresholds * np.ones(output_shape)
        dense_weights = np.array(self.dense_weights)
        dense_ans = np.matmul(dense_weights, dense_input.flatten()).reshape(dense_thresholds.shape)
        expected_spikes = self.get_expected_spikes(dense_ans, dense_thresholds)

        result = self.run_dense_layer_2d(dense_input.shape)
        
        # get output positions in result
        output_positions = self.get_output_spike_positions()         
        output_mask = self.get_output_mask(output_positions, result)       

        calculated_spikes = list(result[output_mask].to_numpy()[:, 0])
        assert expected_spikes == calculated_spikes    

    @pytest.mark.parametrize("nWeights,weight_shape",[(10,(3,3)), (10, (9,)), (5, (2,2))])
    def test_dense_layer_weights_shape(self, nWeights, weight_shape):
        self.basep = 3
        self.bits = 3
        self.filters = [[2, 3],[4,5]]
        self.mode = "full"
        self.pvector = [[1,1,4,6],[6,2,4,2],[2,3,5,4],[6,1,6,3]]

        self.pool_size = 2
        self.pool_strides = 2
        self.pool_thresholds = 0.9
        self.pool_method = "max"

        self.dense_thresholds = [[6.9, 18.9],[30.9, 43.0]]
        self.dense_weights = np.arange(1,nWeights,dtype=float).reshape(weight_shape)

        self.conv_ans = convolve2d(self.pvector, self.filters, mode=self.mode)
        self.pixel_shape = self.conv_ans.shape
        self.pixel_dim1 = self.pixel_shape[0]
        self.pixel_dim2 = self.pixel_shape[1]
        self.conv_spike_pos = np.zeros(self.conv_ans.shape)
        self.conv_spike_pos[[0,0,1,1,2,3,4],[1,3,0,1,2,3,4]] = 0.1

        self.conv_thresholds = self.conv_ans - self.conv_spike_pos
        pool_input = self.conv_ans > self.conv_thresholds
        
        # Check calculations
        dense_input = self.get_expected_pooling_answer(pool_input)

        # dense_thresholds = np.array(self.dense_thresholds)
        # dense_weights = np.array(self.dense_weights)
        # dense_ans = np.matmul(dense_weights, dense_input.flatten()).reshape(dense_thresholds.shape)
        # expected_spikes = self.get_expected_spikes(dense_ans, dense_thresholds)

        with pytest.raises(ValueError):
            self.run_dense_layer_2d(dense_input.shape)

    @pytest.mark.parametrize("weight_shape", [(4,4)])
    def test_dense_layer_scalar_weights(self, weight_shape):
        self.basep = 3
        self.bits = 3
        self.filters = [[2, 3],[4,5]]
        self.mode = "full"
        self.pvector = [[1,1,4,6],[6,2,4,2],[2,3,5,4],[6,1,6,3]]

        self.pool_size = 2
        self.pool_strides = 2
        self.pool_thresholds = 0.9
        self.pool_method = "max"

        self.dense_thresholds = [[6.9, 18.9],[30.9, 43.0]]
        self.dense_weights = 2.0

        self.conv_ans = convolve2d(self.pvector, self.filters, mode=self.mode)
        self.pixel_shape = self.conv_ans.shape
        self.pixel_dim1 = self.pixel_shape[0]
        self.pixel_dim2 = self.pixel_shape[1]
        self.conv_spike_pos = np.zeros(self.conv_ans.shape)
        self.conv_spike_pos[[0,0,1,1,2,3,4],[1,3,0,1,2,3,4]] = 0.1

        self.conv_thresholds = self.conv_ans - self.conv_spike_pos
        pool_input = self.conv_ans > self.conv_thresholds
        
        # Check calculations
        dense_input = self.get_expected_pooling_answer(pool_input)

        dense_thresholds = np.array(self.dense_thresholds)
        dense_weights = self.dense_weights * np.ones(weight_shape)
        dense_ans = np.matmul(dense_weights, dense_input.flatten()).reshape(dense_thresholds.shape)
        expected_spikes = self.get_expected_spikes(dense_ans, dense_thresholds)

        result = self.run_dense_layer_2d(dense_input.shape)
        
        # get output positions in result
        output_positions = self.get_output_spike_positions()         
        output_mask = self.get_output_mask(output_positions, result)       

        calculated_spikes = list(result[output_mask].to_numpy()[:, 0])
        assert expected_spikes == calculated_spikes

    @pytest.mark.xfail(reason="Not implemented.")
    def test_dense_layer_with_different_input_output_shapes(self):
        #TODO: Unit test that uses different input/output neuron shapes
        assert False

    def get_output_neuron_numbers(self):
        neuron_numbers = []
        for key in self.graph.nodes.keys():
            if key.startswith('dense_d'):
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

    def run_dense_layer_2d(self, output_shape):
        scaffold = Scaffold()
        scaffold.add_brick(BaseP_Input(np.array([self.pvector]),p=self.basep,bits=self.bits,collapse_binary=False,name="Input0",time_dimension=False),"input")
        scaffold.add_brick(convolution_2d(self.pvector,self.filters,self.conv_thresholds,self.basep,self.bits,name="convolution_",mode=self.mode),[(0, 0)],output=True)
        scaffold.add_brick(pooling_2d(self.pool_size,self.pool_strides,thresholds=self.pool_thresholds,name="pool_",method=self.pool_method),[(1,0)],output=True)
        scaffold.add_brick(dense_layer_2d(output_shape,weights=self.dense_weights,thresholds=self.dense_thresholds,name="dense_"),[(2,0)],output=True)

        self.graph = scaffold.lay_bricks()
        scaffold.summary(verbose=1)
        backend = snn_Backend()
        backend_args = {}
        backend.compile(scaffold, backend_args)
        result = backend.run(5)
        return result
    
    def get_stride_positions(self, pixel_dim):
        return np.arange(0, pixel_dim, self.pool_strides, dtype=int)
    
    def get_output_size(self, pixel_dim):
        return int(np.floor(1.0 + (np.float64(pixel_dim) - self.pool_size)/self.pool_strides))

    def get_output_shape(self):
        return (self.get_output_size(self.pixel_dim1), self.get_output_size(self.pixel_dim2))
    
    def get_expected_pooling_answer(self, pool_input):
        row_stride_positions = self.get_stride_positions(self.pixel_dim1)
        col_stride_positions = self.get_stride_positions(self.pixel_dim2)
        output_shape = self.get_output_shape()
        expected = []

        if self.pool_method == "max":
            for row in row_stride_positions[:output_shape[0]]:
                for col in col_stride_positions[:output_shape[1]]:
                    expected.append(np.any(pool_input[row:row+self.pool_size,col:col+self.pool_size]))

            expected = np.reshape(expected, output_shape).astype(int)
            expected = (np.array(expected, dtype=int) > self.pool_thresholds).astype(float)

        elif self.pool_method == "average": #TODO : fix this code. 
            weights = np.ones((self.pool_size,self.pool_size),dtype=float) / float(self.pool_size*self.pool_size)
            for row in row_stride_positions[:output_shape[0]]:
                for col in col_stride_positions[:output_shape[1]]:
                    expected.append(np.dot(weights.flatten(),pool_input[row:row+self.pool_size,col:col+self.pool_size].astype(int).flatten()))

            expected = np.reshape(expected, output_shape).astype(float)
        else:
            print(f"'pool_method' class member variable must be either 'max' or 'average'. But it is {self.pool_method}.")
            raise ValueError("Unrecognized 'pool_method' class member variable.")
            
        return np.array(expected)
    
    def get_expected_spikes(self,expected_ans, thresholds):
        ans = np.array(expected_ans)
        spikes = ans[ ans > thresholds].astype(float)
        return list(3.0 * np.ones(spikes.shape))
