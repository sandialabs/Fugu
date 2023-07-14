import numpy as np
import pytest
from scipy.signal import convolve2d

from fugu.backends import snn_Backend
from fugu.bricks.convolution_bricks import convolution_1d, convolution_2d
from fugu.bricks.dense_bricks import dense_layer_1d, dense_layer_2d
from fugu.bricks.input_bricks import BaseP_Input
from fugu.bricks.pooling_bricks import pooling_1d, pooling_2d
from fugu.scaffold import Scaffold

import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
import whetstone

def mock_keras_model(filename):
    '''
        Monkeypatch (i.e., mock) for a keras model.
    '''
    input_shape = (28,28,3)
    model = Sequential()
    model.add(Conv2D(3, (7, 7), padding='same', activation=None, use_bias=True, input_shape=input_shape))
    return model

class Test_Whetstone_2_Fugu_Normalization_Off:
    @pytest.mark.xfail(reason="Not implemented.")
    def test_layers(self):
        model = whetstone.utils.load_model("tests/unit/utils/data/model_adaptive_mnist_normalization_off.keras")

        assert False

    @pytest.mark.xfail(reason="Not implemented.")
    def test_mock_keras_model(self, monkeypatch):
        monkeypatch.setattr("whetstone.utils.load_model", mock_keras_model)
        model = whetstone.utils.load_model("model.keras")

        assert False

class Test_Whetstone_2_Fugu_Normalization_On:
    @pytest.mark.xfail(reason="Not implemented.")
    def test_layers(self):
        assert False