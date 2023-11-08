import numpy as np
import pytest

from fugu.backends import snn_Backend
from fugu.bricks.keras_convolution_bricks import keras_convolution_2d as convolution_2d
from fugu.bricks.dense_bricks import dense_layer_1d, dense_layer_2d
from fugu.bricks.input_bricks import BaseP_Input, Vector_Input
from fugu.bricks.pooling_bricks import pooling_1d, pooling_2d
from fugu.scaffold import Scaffold
from fugu.utils.whetstone import whetstone_2_fugu

import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, BatchNormalization
from whetstone.layers import Spiking_BRelu
from whetstone.utils import layer_utils

class Test_Whetstone_2_Fugu_DenseLayer:
    @pytest.mark.xfail(reason="Not implemented.")
    def test_dense_layer(self):
        assert False