# isort: skip_file
# fmt: off
import numpy as np
import pytest

from fugu.backends import snn_Backend
from fugu.bricks.keras_convolution_bricks import keras_convolution_2d as convolution_2d
from fugu.bricks.dense_bricks import dense_layer_1d, dense_layer_2d
from fugu.bricks.input_bricks import BaseP_Input, Vector_Input
from fugu.bricks.pooling_bricks import pooling_1d, pooling_2d
from fugu.scaffold import Scaffold
from fugu.utils.keras_helpers import keras_convolve2d, keras_convolve2d_4dinput, generate_keras_kernel, generate_mock_image, keras_convolution2d_output_shape, keras_convolution2d_output_shape_4dinput
from fugu.utils.whetstone import whetstone_2_fugu

import pandas as pd
from tensorflow.keras import Model, initializers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, BatchNormalization

Spiking_BRelu = pytest.importorskip("whetstone.layers", reason=f"Whetstone package not installed. Skipping test file {__file__} because of module dependency.").Spiking_BRelu
layer_utils = pytest.importorskip("whetstone.utils", reason=f"Whetstone package not installed. Skipping test file {__file__} because of module dependency.").layer_utils

class Test_Whetstone_2_Fugu_DenseLayer:
    @pytest.mark.xfail(reason="Not implemented.")
    def test_simple_keras_dense_layer(self):
        assert False

    @pytest.mark.xfail(reason="Not implemented.")
    def test_keras_dense_layer(self):
        assert False

    @pytest.mark.xfail(reason="Not implemented.")
    def test_keras_dense_layer_with_biases_without_activation(self):
        assert False

    @pytest.mark.xfail(reason="Not implemented.")
    def test_keras_dense_layer_without_biases_without_activation(self):
        assert False

    @pytest.mark.xfail(reason="Not implemented.")
    def test_keras_dense_layer_with_biases_with_activation(self):
        assert False

    @pytest.mark.xfail(reason="Not implemented.")
    def test_keras_dense_layer_without_biases_with_activation(self):
        assert False
