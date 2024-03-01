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

@pytest.mark.keras
@pytest.mark.keras_pooling
class Test_KerasPooling2D:

    def setup_method(self):
        self.basep = 4
        self.bits = 4

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
        calculated_spike_count = len(self.get_pooling_neurons_result_only(result).index)
        assert expected_spike_count == calculated_spike_count

    def test_explicit_max_pooling_same_mode_strides_12(self):
        convo_obj = ConvolutionParams(biases=np.array([-471., -1207., -1943.]))
        pool_obj = PoolingParams(convo_obj, pool_strides=(1,2), pool_padding="same")

        expected_spike_count = (pool_obj.pool_answer > pool_obj.pool_thresholds).sum().astype(int)

        result = self.run_pooling_2d(convo_obj,pool_obj)
        calculated_spike_count = len(self.get_pooling_neurons_result_only(result).index)
        assert expected_spike_count == calculated_spike_count

    def test_explicit_max_pooling_same_mode_strides_21(self):
        convo_obj = ConvolutionParams(biases=np.array([-471., -1207., -1943.]))
        pool_obj = PoolingParams(convo_obj, pool_strides=(2,1), pool_padding="same")

        expected_spike_count = (pool_obj.pool_answer > pool_obj.pool_thresholds).sum().astype(int)

        result = self.run_pooling_2d(convo_obj,pool_obj)
        calculated_spike_count = len(self.get_pooling_neurons_result_only(result).index)
        assert expected_spike_count == calculated_spike_count

    def test_explicit_max_pooling_same_mode_strides_22(self):
        convo_obj = ConvolutionParams(biases=np.array([-471., -1207., -1943.]))
        pool_obj = PoolingParams(convo_obj, pool_strides=(2,2), pool_padding="same")

        expected_spike_count = (pool_obj.pool_answer > pool_obj.pool_thresholds).sum().astype(int)

        result = self.run_pooling_2d(convo_obj,pool_obj)
        calculated_spike_count = len(self.get_pooling_neurons_result_only(result).index)
        assert expected_spike_count == calculated_spike_count

    def test_explicit_max_pooling_valid_mode_strides_11(self):
        convo_obj = ConvolutionParams(biases=np.array([-471., -1207., -1943.]))
        pool_obj = PoolingParams(convo_obj, pool_strides=(1,1), pool_padding="valid")

        expected_spike_count = (pool_obj.pool_answer > pool_obj.pool_thresholds).sum().astype(int)

        result = self.run_pooling_2d(convo_obj,pool_obj)
        calculated_spike_count = len(self.get_pooling_neurons_result_only(result).index)
        assert expected_spike_count == calculated_spike_count

    def test_explicit_max_pooling_valid_mode_strides_12(self):
        convo_obj = ConvolutionParams(biases=np.array([-471., -1207., -1943.]))
        pool_obj = PoolingParams(convo_obj, pool_strides=(1,2), pool_padding="valid")

        expected_spike_count = (pool_obj.pool_answer > pool_obj.pool_thresholds).sum().astype(int)

        result = self.run_pooling_2d(convo_obj,pool_obj)
        calculated_spike_count = len(self.get_pooling_neurons_result_only(result).index)
        assert expected_spike_count == calculated_spike_count

    def test_explicit_max_pooling_valid_mode_strides_21(self):
        convo_obj = ConvolutionParams(biases=np.array([-471., -1207., -1943.]))
        pool_obj = PoolingParams(convo_obj, pool_strides=(2,1), pool_padding="valid")

        expected_spike_count = (pool_obj.pool_answer > pool_obj.pool_thresholds).sum().astype(int)

        result = self.run_pooling_2d(convo_obj,pool_obj)
        calculated_spike_count = len(self.get_pooling_neurons_result_only(result).index)
        assert expected_spike_count == calculated_spike_count

    def test_explicit_max_pooling_valid_mode_strides_22(self):
        convo_obj = ConvolutionParams(biases=np.array([-471., -1207., -1943.]))
        pool_obj = PoolingParams(convo_obj, pool_strides=(2,2), pool_padding="valid")

        expected_spike_count = (pool_obj.pool_answer > pool_obj.pool_thresholds).sum().astype(int)

        result = self.run_pooling_2d(convo_obj,pool_obj)
        calculated_spike_count = len(self.get_pooling_neurons_result_only(result).index)
        assert expected_spike_count == calculated_spike_count

    def test_explicit_average_pooling_same_mode_strides_11(self):
        convo_obj = ConvolutionParams(biases=np.array([-471., -1207., -1943.]))
        pool_obj = PoolingParams(convo_obj, pool_method="average", pool_strides=(1,1), pool_padding="same")

        expected_spike_count = (pool_obj.pool_answer > pool_obj.pool_thresholds).sum().astype(int)

        result = self.run_pooling_2d(convo_obj,pool_obj)
        calculated_spike_count = len(self.get_pooling_neurons_result_only(result).index)
        assert expected_spike_count == calculated_spike_count

    def test_explicit_average_pooling_same_mode_strides_12(self):
        convo_obj = ConvolutionParams(biases=np.array([-471., -1207., -1943.]))
        pool_obj = PoolingParams(convo_obj, pool_method="average", pool_strides=(1,2), pool_padding="same")

        expected_spike_count = (pool_obj.pool_answer > pool_obj.pool_thresholds).sum().astype(int)

        result = self.run_pooling_2d(convo_obj,pool_obj)
        calculated_spike_count = len(self.get_pooling_neurons_result_only(result).index)
        assert expected_spike_count == calculated_spike_count

    def test_explicit_average_pooling_same_mode_strides_21(self):
        convo_obj = ConvolutionParams(biases=np.array([-471., -1207., -1943.]))
        pool_obj = PoolingParams(convo_obj, pool_method="average", pool_strides=(2,1), pool_padding="same")

        expected_spike_count = (pool_obj.pool_answer > pool_obj.pool_thresholds).sum().astype(int)

        result = self.run_pooling_2d(convo_obj,pool_obj)
        calculated_spike_count = len(self.get_pooling_neurons_result_only(result).index)
        assert expected_spike_count == calculated_spike_count

    def test_explicit_average_pooling_same_mode_strides_22(self):
        convo_obj = ConvolutionParams(biases=np.array([-471., -1207., -1943.]))
        pool_obj = PoolingParams(convo_obj, pool_method="average", pool_strides=(2,2), pool_padding="same")

        expected_spike_count = (pool_obj.pool_answer > pool_obj.pool_thresholds).sum().astype(int)

        result = self.run_pooling_2d(convo_obj,pool_obj)
        calculated_spike_count = len(self.get_pooling_neurons_result_only(result).index)
        assert expected_spike_count == calculated_spike_count

    def test_explicit_average_pooling_valid_mode_strides_11(self):
        convo_obj = ConvolutionParams(biases=np.array([-471., -1207., -1943.]))
        pool_obj = PoolingParams(convo_obj, pool_method="average", pool_strides=(1,1), pool_padding="valid")

        expected_spike_count = (pool_obj.pool_answer > pool_obj.pool_thresholds).sum().astype(int)

        result = self.run_pooling_2d(convo_obj,pool_obj)
        calculated_spike_count = len(self.get_pooling_neurons_result_only(result).index)
        assert expected_spike_count == calculated_spike_count

    def test_explicit_average_pooling_valid_mode_strides_12(self):
        convo_obj = ConvolutionParams(biases=np.array([-471., -1207., -1943.]))
        pool_obj = PoolingParams(convo_obj, pool_method="average", pool_strides=(1,2), pool_padding="valid")

        expected_spike_count = (pool_obj.pool_answer > pool_obj.pool_thresholds).sum().astype(int)

        result = self.run_pooling_2d(convo_obj,pool_obj)
        calculated_spike_count = len(self.get_pooling_neurons_result_only(result).index)
        assert expected_spike_count == calculated_spike_count

    def test_explicit_average_pooling_valid_mode_strides_21(self):
        convo_obj = ConvolutionParams(biases=np.array([-471., -1207., -1943.]))
        pool_obj = PoolingParams(convo_obj, pool_method="average", pool_strides=(2,1), pool_padding="valid")

        expected_spike_count = (pool_obj.pool_answer > pool_obj.pool_thresholds).sum().astype(int)

        result = self.run_pooling_2d(convo_obj,pool_obj)
        calculated_spike_count = len(self.get_pooling_neurons_result_only(result).index)
        assert expected_spike_count == calculated_spike_count

    def test_explicit_average_pooling_valid_mode_strides_22(self):
        convo_obj = ConvolutionParams(biases=np.array([-471., -1207., -1943.]))
        pool_obj = PoolingParams(convo_obj, pool_method="average", pool_strides=(2,2), pool_padding="valid")

        expected_spike_count = (pool_obj.pool_answer > pool_obj.pool_thresholds).sum().astype(int)

        result = self.run_pooling_2d(convo_obj,pool_obj)
        calculated_spike_count = len(self.get_pooling_neurons_result_only(result).index)
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
        calculated_spike_count = len(self.get_pooling_neurons_result_only(result).index)
        assert expected_spike_count == calculated_spike_count

    def test_pooling_one_off1(self):
        self.basep = 4
        self.bits = 3

        convo_obj = ConvolutionParams(image_height=5, image_width=5, nChannels=2, kernel_height=2, kernel_width=2, nFilters=3, biases=None)
        convo_obj.biases = convo_obj.get_random_biases_within_answer_range()
        convo_obj._set_convolution_answer_boolean()

        pool_obj = PoolingParams(convo_obj, pool_size=(2,3), pool_strides=(1,1), pool_padding="same", pool_method="max")
        expected_spike_count = (pool_obj.pool_answer > pool_obj.pool_thresholds).sum().astype(int)

        result = self.run_pooling_2d(convo_obj,pool_obj)
        calculated_spike_count = len(self.get_pooling_neurons_result_only(result).index)
        assert expected_spike_count == calculated_spike_count

    def test_pooling_one_off2(self):
        self.basep = 4
        self.bits = 3

        convo_obj = ConvolutionParams(image_height=5, image_width=5, nChannels=2, kernel_height=2, kernel_width=2, nFilters=3, biases=None)
        convo_obj.biases = convo_obj.get_random_biases_within_answer_range()
        convo_obj._set_convolution_answer_boolean()

        pool_obj = PoolingParams(convo_obj, pool_size=(2,1), pool_strides=(1,3), pool_padding="same", pool_method="max")
        expected_spike_count = (pool_obj.pool_answer > pool_obj.pool_thresholds).sum().astype(int)

        result = self.run_pooling_2d(convo_obj,pool_obj)
        calculated_spike_count = len(self.get_pooling_neurons_result_only(result).index)
        assert expected_spike_count == calculated_spike_count

    def test_pooling_one_off3(self):
        self.basep = 4
        self.bits = 4
        convo_obj = ConvolutionParams(nFilters=4,biases=np.array([-471., -1207., -1943., 500.]))
        pool_obj = PoolingParams(convo_obj, pool_strides=(1,1), pool_padding="same")
        expected_spike_count = (pool_obj.pool_answer > pool_obj.pool_thresholds).sum().astype(int)

        result = self.run_pooling_2d(convo_obj,pool_obj)
        calculated_spike_count = len(self.get_pooling_neurons_result_only(result).index)
        assert expected_spike_count == calculated_spike_count

    @pytest.mark.xfail(reason="Not implemented.")
    #TODO: Implement handling of data_format="channels_last" and data_format="channels_first"
    def test_data_format_channels_last(self):
        assert False

    def get_neuron_numbers(self, name_prefix):
        neuron_numbers = []
        for key in self.graph.nodes.keys():
            if key.startswith(name_prefix):
                neuron_numbers.append(self.graph.nodes[key]['neuron_number'])

        return np.array(neuron_numbers)

    def get_pooling_neurons_result_only(self, result):
        pool_neuron_numbers = self.get_neuron_numbers('pool_p')
        sub_result = result[result['neuron_number'].isin(pool_neuron_numbers)]
        return sub_result

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
