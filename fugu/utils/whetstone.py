# isort: skip_file
import numpy as np

from fugu.bricks.keras_convolution_bricks import keras_convolution_2d as convolution_2d
from fugu.bricks.input_bricks import BaseP_Input
from fugu.bricks.pooling_bricks import pooling_1d, pooling_2d
from fugu.bricks.dense_bricks import dense_layer_2d
from fugu.scaffold import Scaffold

from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from whetstone.layers import Spiking_BRelu
from whetstone.utils.export_utils import copy_remove_batchnorm

# Turn off black formatting for this file
# fmt: off

def whetstone_2_fugu(keras_model, basep, bits, scaffold=None):
    '''
    
    '''
    if scaffold is None:
        scaffold = Scaffold()


    # model = copy_remove_batchnorm(keras_model)
    model = keras_model
    layerID = 0
    for idx, layer in enumerate(model.layers):
        if type(layer) is Conv2D:
            # TODO: Add capability to handle "data_format='channels_first'". Current implementation assumes data_format='channels_last'.
            # need pvector shape, filters, thresholds, basep, bits, and mode
            input_shape = layer.input_shape
            output_shape = layer.output_shape
            mode = layer.padding
            kernel = layer.get_weights()[0]
            bias = layer.get_weights()[1]
            strides = layer.strides
            # Add a brick for each channel
            for channel in np.arange(input_shape[-1]):
                for filter in np.arange(layer.filters):
                    print(f"Conv2D:: Channel: {channel} Filter: {filter}")
                    scaffold.add_brick(convolution_2d(input_shape[1:-1],np.flip(kernel[:,:,channel,filter]),1.0,basep,bits,name=f"convolution_layer_{layerID}",mode=mode,strides=strides),[(layerID, 0)],output=True)
                    layerID += 1

        if type(layer) is Spiking_BRelu:
            pass

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