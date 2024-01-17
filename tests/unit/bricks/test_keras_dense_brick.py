# isort: skip_file
# fmt: off
import numpy as np
import pytest
from contextlib import nullcontext as does_not_raise

from fugu.backends import snn_Backend
from fugu.bricks import Vector_Input, Mock_Brick
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

    @pytest.mark.parametrize("weights,output_units,expectation", [([1.0,1.0],27,pytest.raises(ValueError)),((1.0,1.0),27,pytest.raises(ValueError)),(1.0,27,does_not_raise()),(1.0*np.ones((1,3,3,3)),27,pytest.raises(ValueError)),(1.0*np.ones((3,27)),27,does_not_raise()),
                                                                  ([1.0,1.0],12,pytest.raises(ValueError)),((1.0,1.0),12,pytest.raises(ValueError)),(1.0,12,does_not_raise()),(1.0*np.ones((1,3,3,3)),12,pytest.raises(ValueError)),(1.0*np.ones((3,12)),12,does_not_raise()),
                                                                  (1.0*np.ones((27,12)),12,pytest.raises(ValueError))])
    def test_input_weights(self, weights, output_units, expectation):
        convo_obj = ConvolutionParams(biases=np.array([-471., -1207., -1943.]))
        pool_obj = PoolingParams(convo_obj)
        dense_obj = DenseParams(pool_obj, output_units)
        dense_obj.weights = weights
        # dense_obj._set_output_shape(output_shape)
        dense_obj._set_thresholds(0.9)
        with expectation:
            self.run_dense_2d(convo_obj,pool_obj,dense_obj)

    @pytest.mark.parametrize("thresholds,output_units,expectation", [([0.9,0.9],27,pytest.raises(ValueError)),((0.9,0.9),27,pytest.raises(ValueError)),(0.9,27,does_not_raise()),(0.9*np.ones((1,3,3,3)),27,pytest.raises(ValueError)),
                                                                     ([0.9,0.9],12,pytest.raises(ValueError)),((0.9,0.9),12,pytest.raises(ValueError)),(0.9,12,does_not_raise()),(0.9*np.ones((1,2,2,3)),12,pytest.raises(ValueError))])
    def test_input_thresholds(self, thresholds, output_units, expectation):
        convo_obj = ConvolutionParams(biases=np.array([-471., -1207., -1943.]))
        pool_obj = PoolingParams(convo_obj)
        dense_obj = DenseParams(pool_obj, output_units)
        dense_obj.thresholds = thresholds
        dense_obj._set_weights(1.0)
        with expectation:
            self.run_dense_2d(convo_obj,pool_obj,dense_obj)

    def test_simple_explicit_dense_layer_example_1(self):
        convo_obj = ConvolutionParams(nFilters=4,biases=np.array([-471., -1207., -1943., -500.]))
        pool_obj = PoolingParams(convo_obj, pool_strides=(1,1), pool_padding="same")
        dense_obj = DenseParams(pool_obj, output_units=16)

        dense_input = pool_obj.pool_answer.astype(int)
        dense_answer = dense_obj.get_dense_answer(dense_input)
        expected_spike_count = (dense_obj.dense_answer + dense_obj.biases > dense_obj.thresholds).sum().astype(int)

        result = self.run_dense_2d(convo_obj,pool_obj,dense_obj)
        calculated_spike_count = len(result[result['time'] > 2].index)
        assert calculated_spike_count == expected_spike_count

    @pytest.mark.parametrize("bias",[0.0,-20.0,-21.0, None])
    def test_dense_brick_biases(self, bias):
        convo_obj = ConvolutionParams(nFilters=4,biases=np.array([-471., -1207., -1943., -500.]))
        pool_obj = PoolingParams(convo_obj, pool_strides=(1,1), pool_padding="same")
        dense_obj = DenseParams(pool_obj, output_units=3, biases=bias)

        dense_input = pool_obj.pool_answer.astype(int)
        dense_answer = dense_obj.get_dense_answer(dense_input)
        expected_spike_count = (dense_obj.dense_answer + dense_obj.biases > dense_obj.thresholds).sum().astype(int)

        result = self.run_dense_2d(convo_obj,pool_obj,dense_obj)
        calculated_spike_count = len(result[result['time'] > 2].index)
        assert calculated_spike_count == expected_spike_count

    def test_mock_brick(self):

        nFilters = 4
        output_units = np.prod((1,2,2,nFilters))
        convo_obj = ConvolutionParams(nFilters=nFilters,biases=np.array([-471., -1207., -1943., 500.]))
        pool_obj = PoolingParams(convo_obj, pool_strides=(1,1), pool_padding="same")
        dense_obj = DenseParams(pool_obj, output_units)

        vector_input = Vector_Input(pool_obj.pool_answer,name="pool_",time_dimension=False)
        mock_input = Mock_Brick(vector_input,{'pooling_output_shape': pool_obj.pool_answer.shape})

        scaffold = Scaffold()
        scaffold.add_brick(mock_input,"input")
        scaffold.add_brick(keras_dense_2d(units=output_units,weights=1.0,thresholds=0.9,data_format="channels_last",name="dense_"),[(0,0)],output=True)

        graph = scaffold.lay_bricks()
        scaffold.summary(verbose=1)
        backend = snn_Backend()
        backend_args = {}
        backend.compile(scaffold, backend_args)
        result = backend.run(5)

        calculated_spike_count = len(result[result['time'] > 0].index)
        expected_spike_count = (dense_obj.dense_answer > dense_obj.thresholds).sum().astype(int)
        assert expected_spike_count == calculated_spike_count

    @pytest.mark.xfail(reason="Not implemented.")
    def test_something(self):
        assert False

    def get_neuron_numbers(self, name_prefix):
        neuron_numbers = []
        for key in self.graph.nodes.keys():
            if key.startswith(name_prefix):
                neuron_numbers.append(self.graph.nodes[key]['neuron_number'])

        return np.array(neuron_numbers)

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

    def run_dense_2d(self, convo_obj, pool_obj, dense_obj):
        scaffold = Scaffold()
        scaffold.add_brick(BaseP_Input(convo_obj.mock_image,p=self.basep,bits=self.bits,collapse_binary=False,name="I",time_dimension=False),"input")
        scaffold.add_brick(keras_convolution_2d(convo_obj.input_shape,convo_obj.filters,convo_obj.thresholds,self.basep,self.bits,name="convolution_",mode=convo_obj.mode,strides=convo_obj.strides,biases=convo_obj.biases),[(0, 0)],output=True)
        scaffold.add_brick(keras_pooling_2d(pool_obj.pool_size,pool_obj.pool_strides,thresholds=pool_obj.pool_thresholds,name="pool_",padding=pool_obj.pool_padding,method=pool_obj.pool_method),[(1,0)],output=True)
        scaffold.add_brick(keras_dense_2d(dense_obj.output_units,dense_obj.weights,dense_obj.thresholds,data_format=dense_obj.data_format,name="dense_",biases=dense_obj.biases),[(2,0)],output=True)

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
    
    def get_expected_spikes(self,expected_ans, thresholds):
        ans = np.array(expected_ans)
        spikes = ans[ ans > thresholds].astype(float)
        return list(3.0 * np.ones(spikes.shape))

    def get_result_spike_count(self, result):
        output_positions = self.get_output_spike_positions()
        output_mask = self.get_output_mask(output_positions, result)

        calculated_spikes = list(result[output_mask].to_numpy()[:, 0])
        return np.array(calculated_spikes).sum()

    def get_dense_neurons_result_only(self, result):
        dense_neuron_numbers = self.get_neuron_numbers('dense_d')
        sub_result = result[result['neuron_number'].isin(dense_neuron_numbers)]
        return sub_result
