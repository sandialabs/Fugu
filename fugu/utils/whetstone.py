import numpy as np
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

model = whetstone.utils.load_model('tests/unit/utils/data/model_adaptive_mnist_normalization_off.keras')

def whetstone_2_fugu(keras_model, basep, bits, scaffold=None):
    '''
    
    '''
    if scaffold is None:
        scaffold = Scaffold()


    model = whetstone.utils.export_utils.copy_remove_batchnorm(keras_model)
    layerID = 0
    for idx, layer in enumerate(model.layers):
        if type(layer) is Conv2D:
            # need pvector shape, filters, thresholds, basep, bits, and mode
            input_shape = layer.input_shape
            output_shape = layer.output_shape
            padding = layer.padding
            weights = layer.get_weights()[0]
            bias = layer.get_weights()[1]
            strides = layer.strides
            # Add a brick for each channel
            for channel in np.arange(input_shape[-1]):
                #TODO: update convolution brick to use pvector shape, instead of pvector, as an input parameter to the brick. Note, the convolution brick assumes
                # the strides are 1 in each direction.
                scaffold.add_brick(convolution_2d(np.zeros(input_shape[1:-1]),weights,bias,basep,bits,name=f"convolution_layer_{layerID}",mode=padding),[(layerID, 0)],output=True)
                layerID += 1

        if type(layer) is MaxPooling2D:
            # need pool size, strides, thresholds, and method
            pool_size = layer.pool_size
            input_shape = layer.input_shape
            output_shape = layer.output_shape
            padding = layer.padding
            strides = layer.strides
            # Add a brick for each channel
            for channel in np.arange(input_shape[-1]):
                # TODO: update pooling brick to accept 2D tuples for pool size and strides. For now, the brick assumes the pool size/strides is constant in both directions
                scaffold.add_brick(pooling_2d(pool_size[0],strides[0],name=f"pool_layer_{layerID}",method="max"),[(layerID,0)],output=True)
                layerID += 1

        if type(layer) is Dense:
            # need output shape, weights, thresholds 
            output_shape = layer.output_shape
            weights = layer.weights[0]
            bias = layer.weights[1]
            scaffold.add_brick(dense_layer_2d(output_shape,weights=weights,thresholds=bias,name=f"dense_layer_{layerID}"),[(layerID,0)],output=True)
            layerID += 1
    
    return scaffold