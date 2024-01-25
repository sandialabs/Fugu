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
from ..helpers import ConvolutionParams, PoolingParams, DenseParams, IntegerSequence, ArraySequence

@pytest.mark.keras
@pytest.mark.keras_dense
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

    @pytest.mark.parametrize("bias",[0.0,-3.0,-20.0,-21.0, None])
    def test_dense_brick_biases(self, bias):
        convo_obj = ConvolutionParams(nFilters=4,biases=np.array([-471., -1207., -1943., -500.]))
        pool_obj = PoolingParams(convo_obj, pool_strides=(1,1), pool_padding="same")

        units = 2
        myweights = np.arange(1,units*pool_obj.nChannels+1).reshape(pool_obj.nChannels,units)
        dense_obj = DenseParams(pool_obj, output_units=units, weights=myweights, thresholds=0.5,biases=bias)

        dense_input = pool_obj.pool_answer.astype(int)
        dense_answer = dense_obj.get_dense_answer(dense_input)
        expected_spike_count = (dense_obj.dense_answer + dense_obj.biases > dense_obj.thresholds).sum().astype(int)

        result = self.run_dense_2d(convo_obj,pool_obj,dense_obj)
        calculated_spike_count = len(result[result['time'] > 2].index)
        assert calculated_spike_count == expected_spike_count

    def test_dense_brick_biases2(self):
        batch_size, height, width, nChannels = 1, 3, 3, 2
        convo_obj = ConvolutionParams(image_height=height, image_width=width, nChannels=nChannels, nFilters=1, biases=np.array([-471.]))
        pool_obj = PoolingParams(convo_obj,pool_size=(2,2), pool_strides=(1,1), pool_padding="same", pool_method='max')

        # Set dense Layer Parameters
        units = 2
        weights = np.arange(1,units*pool_obj.nChannels+1).reshape(pool_obj.nChannels,units).astype(float)
        biases = [1e9, 0.0]
        thresholds = 1.0
        # thresholds = generate_mock_image(height,width,units).astype(float)
        # thresholds[:] = 0.
        # thresholds[0,:2,:2,:] = [[[1.0,2.0],[0.9,2.0]],[[1.0,2.0],[1.0,1.9]]]
        # thresholds = np.arange(1,units*np.prod(pool_obj.output_shape)+1).reshape(*np.array(pool_obj.output_shape)[:3],units)
        dense_obj = DenseParams(pool_obj, output_units=units, weights=weights, thresholds=thresholds,biases=biases)

        dense_input = (pool_obj.pool_answer > pool_obj.pool_thresholds).astype(int)
        dense_answer = dense_obj.get_dense_answer(dense_input)
        expected_spike_count = (dense_obj.dense_answer + dense_obj.biases > dense_obj.thresholds).sum().astype(int)

        result = self.run_dense_2d(convo_obj,pool_obj,dense_obj)
        calculated_spike_count = len(result[result['time'] > 2].index)

        # Make Keras Model
        kobj = make_keras_dense_model(units=units,input_shape=tuple(np.array(pool_obj.input_shape)[1:]),weights=weights,biases=biases)

        assert calculated_spike_count == expected_spike_count

    def test_mock_brick1(self):

        nFilters = 4
        output_units = np.prod((1,2,2,nFilters))
        convo_obj = ConvolutionParams(nFilters=nFilters,biases=np.array([-471., -1207., -1943., 500.]))
        pool_obj = PoolingParams(convo_obj, pool_strides=(1,1), pool_padding="same")
        myinputs = np.arange(1,3*3*pool_obj.nChannels+1).reshape((1,3,3,pool_obj.nChannels))
        pool_obj.pool_answer = myinputs

        units = 2
        weights = np.arange(1,units*pool_obj.nChannels+1).reshape(pool_obj.nChannels,units)
        biases = [0., 0.]
        thresholds = 400.0
        dense_obj = DenseParams(pool_obj, output_units=units, weights=weights, thresholds=thresholds, biases=biases)
        vector_input = Vector_Input(pool_obj.pool_answer,name="pool_",time_dimension=False)
        mock_input = Mock_Brick(vector_input,{'pooling_output_shape': pool_obj.pool_answer.shape})

        scaffold = Scaffold()
        scaffold.add_brick(mock_input,"input")
        scaffold.add_brick(keras_dense_2d(units=dense_obj.output_units,weights=dense_obj.weights,thresholds=thresholds,data_format="channels_last",name="dense_",biases=dense_obj.biases),[(0,0)],output=True)

        graph = scaffold.lay_bricks()
        scaffold.summary(verbose=1)
        backend = snn_Backend()
        backend_args = {}
        backend.compile(scaffold, backend_args)
        result = backend.run(5)

        calculated_spike_count = len(result[result['time'] > 0].index)
        expected_spike_count = (dense_obj.dense_answer + dense_obj.biases > dense_obj.thresholds).sum().astype(int)
        assert expected_spike_count == calculated_spike_count

    def test_mock_brick2(self):

        # Set "pooling" parameters
        batch_size, height, width, nChannels = 1, 3, 3, 1
        input_shape = (batch_size, height, width, nChannels)
        dense_input = generate_mock_image(height,width,nChannels).reshape(input_shape)
        myobj = CustomParam(input_shape=input_shape,pool_answer=dense_input)

        # Set dense Layer Parameters
        units = 2
        weights = np.arange(1,units*nChannels+1).reshape(nChannels,units)
        biases = [0., 0.]
        thresholds = 1.0
        dense_obj = DenseParams(myobj, output_units=units, weights=weights, thresholds=thresholds, biases=biases)

        # Construct Mock Brick
        mock_brick = Mock_Brick(Vector_Input(dense_input, name="I_", time_dimension=False), 
                                {'pooling_output_shape': dense_input.shape})
        
        # Build Scaffold
        scaffold = Scaffold()
        scaffold.add_brick(mock_brick,"input")
        scaffold.add_brick(keras_dense_2d(units=dense_obj.output_units,weights=dense_obj.weights,thresholds=thresholds,data_format="channels_last",name="dense_",biases=dense_obj.biases),[(0,0)],output=True)

        graph = scaffold.lay_bricks()
        scaffold.summary(verbose=1)
        backend = snn_Backend()
        backend_args = {}
        backend.compile(scaffold, backend_args)
        result = backend.run(5)

        calculated_spike_count = len(result[result['time'] > 0].index)
        expected_spike_count = (dense_obj.dense_answer + dense_obj.biases > dense_obj.thresholds).sum().astype(int)

        # Make Keras Model
        kobj = make_keras_dense_model(units=units,input_shape=input_shape,weights=weights,biases=biases)
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

class CustomParam():
    def __init__(self,input_shape,pool_answer):
        self.input_shape = input_shape
        self.pool_answer = pool_answer

def make_keras_dense_model(units, input_shape, weights=None, biases=None):
    from tensorflow.keras import Model, initializers
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense

    class MyClass():
        pass

    if len(input_shape) == 4:
        batch_size, height, width, nChannels = input_shape
        input_shape = (height, width, nChannels)
    elif len(input_shape) == 3:
        batch_size = 1
        height, width, nChannels = input_shape
        
    if weights is None:
        weights = IntegerSequence()
    elif isinstance(weights, (list,np.ndarray)):
        weights = ArraySequence(weights)
    elif isinstance(weights,(int,float)):
        weights = initializers.constant(value=weights)

    if biases is None:
        biases = IntegerSequence()
    elif isinstance(biases, (list,np.ndarray)):
        biases = ArraySequence(biases)
    elif isinstance(biases,(int,float)):
        biases = initializers.constant(value=biases)        

    model = Sequential()
    model.add(Dense(units=units, use_bias=True, kernel_initializer=weights, bias_initializer=biases, name="dense", input_shape=input_shape))
    feature_extractor = Model(inputs=model.inputs, outputs=[layer.output for layer in model.layers]) # gives cumalative output of the input through this and previous layers.
    
    dlayer = model.layers[0]
    obj = MyClass()
    obj.feature_extractor = feature_extractor
    obj.dlayer = dlayer
    obj.biases = dlayer.weights[1].numpy()
    obj.weights = dlayer.weights[0].numpy()
    obj.units = dlayer.units
    obj.input_shape = dlayer.input_shape
    obj.output_shape = dlayer.output_shape

    return obj

def mk_something(graph,node_name):
    adict = {node_name: [i[1] for i in graph[node_name]]}
    return adict

def get_node_name_list(prefix,shape):
    batch_size, height, width, nChannels = shape
    node_names_list = [f"{prefix}{channel}{row}{col}" for channel in np.arange(nChannels) for row in np.arange(height) for col in np.arange(width)]
    return node_names_list

def get_edge_connections(graph,prefix,shape):
    node_names = get_node_name_list(prefix,shape)
    adict = {node_name: [i[1] for i in graph.edges(node_name)] for node_name in node_names}
    return adict

def get_edge_connections_inverse(graph,prefix1,shape1,prefix2,shape2):
    node_names1 = get_node_name_list(prefix1,shape1)
    adict1 = get_edge_connections(graph,prefix1,shape1)

    node_names2 = get_node_name_list(prefix2,shape2)
    adict2 = {node_name2: [node_name1 for node_name1 in node_names1 if node_name2 in adict1[node_name1]] for node_name2 in node_names2}
    return adict2

