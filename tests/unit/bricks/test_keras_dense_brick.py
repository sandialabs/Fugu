# isort: skip_file
# fmt: off
import numpy as np
import pytest
from contextlib import nullcontext as does_not_raise

from fugu.backends import snn_Backend
from fugu.bricks.keras_convolution_bricks import keras_convolution_2d_4dinput as keras_convolution_2d
from fugu.bricks.keras_dense_bricks import keras_dense_2d_4dinput as keras_dense_2d
from fugu.bricks.input_bricks import BaseP_Input
from fugu.bricks.keras_pooling_bricks import keras_pooling_2d_4dinput as keras_pooling_2d
from fugu.scaffold import Scaffold

from fugu.utils.keras_helpers import keras_convolve2d, keras_convolve2d_4dinput, generate_keras_kernel, generate_mock_image, keras_convolution2d_output_shape_4dinput
from ..helpers import ConvolutionParams, PoolingParams, DenseParams

class Test_KerasDense2D:

    def setup_method(self):
        self.basep = 4
        self.bits = 4

    @pytest.mark.parametrize("weights,expectation", [([1.0,1.0],pytest.raises(ValueError)),((1.0,1.0),pytest.raises(ValueError)),(1.0,does_not_raise()),(1.0*np.ones((1,3,3,3)),pytest.raises(ValueError)),(1.0*np.ones((9,9,3)),does_not_raise())])
    def test_input_weights(self, weights, expectation):
        convo_obj = ConvolutionParams(biases=np.array([-471., -1207., -1943.]))
        pool_obj = PoolingParams(convo_obj)
        dense_obj = DenseParams(pool_obj)
        dense_obj.weights = weights
        with expectation:
            self.run_dense_2d(convo_obj,pool_obj,dense_obj)

    @pytest.mark.parametrize("thresholds,expectation", [([0.9,0.9],pytest.raises(ValueError)),((0.9,0.9),pytest.raises(ValueError)),(0.9,does_not_raise()),(0.9*np.ones((1,3,3,3)),does_not_raise())])
    def test_input_thresholds(self, thresholds, expectation):
        convo_obj = ConvolutionParams(biases=np.array([-471., -1207., -1943.]))
        pool_obj = PoolingParams(convo_obj)
        dense_obj = DenseParams(pool_obj)
        dense_obj.thresholds = thresholds
        with expectation:
            self.run_dense_2d(convo_obj,pool_obj,dense_obj)

    def test_simple_explicit_dense_layer_example_1(self):
        convo_obj = ConvolutionParams(biases=np.array([-471., -1207., -1943.]))
        pool_obj = PoolingParams(convo_obj, pool_strides=(1,1), pool_padding="same")
        assert False

    @pytest.mark.xfail(reason="Not implemented.")
    def test_something(self):
        assert False


    #TODO: Unit test that checks dense thresholds shape
    #TODO: Unit test that checks dense weights shape
    #TODO: Unit test that test dense brick with a scalar threshold value
    #TODO: Unit test that test dense brick with a scalar weight value
    #TODO: Unit test that uses different input/output neuron shapes

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
        scaffold.add_brick(keras_convolution_2d(self.pshape,self.filters,self.conv_thresholds,self.basep,self.bits,name="convolution_",mode=self.mode),[(0, 0)],output=True)
        scaffold.add_brick(keras_pooling_2d(self.pool_size,self.pool_strides,thresholds=self.pool_thresholds,name="pool_",method=self.pool_method),[(1,0)],output=True)
        scaffold.add_brick(keras_dense_2d(output_shape,weights=self.dense_weights,thresholds=self.dense_thresholds,name="dense_"),[(2,0)],output=True)

        self.graph = scaffold.lay_bricks()
        scaffold.summary(verbose=1)
        backend = snn_Backend()
        backend_args = {}
        backend.compile(scaffold, backend_args)
        result = backend.run(5)
        return result

    def run_dense_2d(self, convo_obj, pool_obj, dense_obj):
        scaffold = Scaffold()
        scaffold.add_brick(BaseP_Input(convo_obj.mock_image,p=self.basep,bits=self.bits,collapse_binary=False,name="I",time_dimension=False),"input")
        scaffold.add_brick(keras_convolution_2d(convo_obj.input_shape,convo_obj.filters,convo_obj.thresholds,self.basep,self.bits,name="convolution_",mode=convo_obj.mode,strides=convo_obj.strides,biases=convo_obj.biases),[(0, 0)],output=True)
        scaffold.add_brick(keras_pooling_2d(pool_obj.pool_size,pool_obj.pool_strides,thresholds=pool_obj.pool_thresholds,name="pool_",padding=pool_obj.pool_padding,method=pool_obj.pool_method),[(1,0)],output=True)
        scaffold.add_brick(keras_dense_2d(dense_obj.output_shape,dense_obj.weights,dense_obj.thresholds,data_format=dense_obj.data_format,name="dense_"),[(2,0)],output=True)

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
