import whetstone
import tensorflow

from fugu.bricks.convolution_bricks import convolution_1d, convolution_2d
from fugu.bricks.input_bricks import BaseP_Input
from fugu.bricks.pooling_bricks import pooling_1d, pooling_2d
from fugu.bricks.dense_bricks import dense_layer_2d
from fugu.scaffold import Scaffold

from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D

tensorflow.keras.layers.Convolution2D
tensorflow.keras.layers.MaxPooling2D

model = whetstone.utils.load_model('C:\\Users\\mkrygie\\Repositories\\Whetstone\\examples\\model_adaptive_mnist.keras')
model.

def whetstone_2_fugu(keras_model,scaffold=None):
    '''
    
    '''
    if scaffold is None:
        scaffold = Scaffold

    model = keras_model
    for idx, layer in enumerate(model.layers):
        if type(layer) is Conv2D:
            # need pvector shape, filters, thresholds, basep, bits, and mode
            input_shape = layer.input_shape
            output_shape = layer.output_shape
            padding = layer.padding
            weights = layer.get_weights()[0]
            bias = layer.get_weights()[1]
            strides = layer.strides
            pass
        if type(layer) is MaxPooling2D:
            # need pool size, strides, thresholds, and method
            pool_size = layer.pool_size
            input_shape = layer.input_shape
            output_shape = layer.output_shape
            padding = layer.padding
            strides = layer.strides
            pass
        if type(layer) is Dense:
            # need output shape, weights, thresholds 
            output_shape = layer.output_shape
            weights = layer.weights[0]
            bias = layer.weights[1]
            pass    
    
    return scaffold