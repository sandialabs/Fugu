import numpy as np
import pytest
from scipy.signal import convolve2d

from fugu.backends import snn_Backend
from fugu.bricks.convolution_bricks import convolution_1d, convolution_2d
from fugu.bricks.dense_bricks import dense_layer_1d, dense_layer_2d
from fugu.bricks.input_bricks import BaseP_Input
from fugu.bricks.pooling_bricks import pooling_1d, pooling_2d
from fugu.scaffold import Scaffold

from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
import whetstone

def mock_keras_model(filename):
    AM = pd.DataFrame(data={'atom': 1, 'type': 1, 'radius': 10.0, 'mass': 1000.0, 'x': 6.0, 'y': 7.0, 'z': 8.0, 'c': 1}, index=[0])
    bounds = np.array([[5., 10.],[5., 10.],[5., 10.]])
    return AM, bounds

class Test_Whetstone_2_Fugu:
    def test_layers(self):
        model = whetstone.utils.load_model("model.keras")

        assert False

    def test_mock_keras_model(self, monkeypatch):
        monkeypatch.setattr("whetstone.utils.load_model", mock_keras_model)
        model = whetstone.utils.load_model("model.keras")

        assert False
