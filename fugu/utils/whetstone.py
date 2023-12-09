# isort: skip_file
import numpy as np

from fugu.bricks.keras_convolution_bricks import keras_convolution_2d_4dinput as convolution_2d
from fugu.bricks.input_bricks import BaseP_Input
from fugu.bricks.pooling_bricks import pooling_1d, pooling_2d
from fugu.bricks.dense_bricks import dense_layer_2d
from fugu.scaffold import Scaffold

from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, BatchNormalization
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
            # TODO: Add capability to handle removal of batchnormalization layer
            # need pvector shape, filters, thresholds, basep, bits, and mode

            # Check if BatchNormalization is the next layer. If so, merge BatchNormalization layer
            # with Convolution2D layer to get the new weights and biases
            next_layer = model.layers[idx + 1] if idx < len(model.layers) - 1 else None
            if type(next_layer) == BatchNormalization:
                kernel, biases = merge_layers(layer,next_layer)
            else:
                kernel = layer.get_weights()[0]
                biases = layer.get_weights()[1]

            if layer.data_format == 'channels_last':
                batch_size = layer.input_shape[:-3]
                if len(batch_size) == 1 and batch_size[0] is None:
                    batch_size = 1
                else:
                    batch_size = batch_size[0]
            elif layer.data_format == 'channels_first':
                #TODO : Handle this scenario later
                pass

            input_shape = tuple([batch_size if value == None else value for value in layer.input_shape])
            output_shape = tuple([batch_size if value == None else value for value in layer.output_shape])
            mode = layer.padding
            strides = layer.strides
            print(f"Conv2D:: LayerID: {layerID+1}")
            scaffold.add_brick(convolution_2d(input_shape,np.flip(kernel,(0,1)),0.5,basep,bits,name=f"convolution_layer{layerID}_",mode=mode,strides=strides,biases=biases),[(layerID, 0)],output=True)
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

# Auxillary/Helper functions
def normalization(batch, batch_normalization_layer):
    gamma, beta, mean, variance = batch_normalization_layer.get_weights()
    epsilon = batch_normalization_layer.epsilon
    return apply_normalization(batch,gamma,beta,mean,variance,epsilon)

def apply_normalization(batch,gamma, beta, moving_mean, moving_var, epsilon):
    return gamma*(batch - moving_mean) / np.sqrt(moving_var+epsilon) + beta

def merge_layers(convolution2d_layer, batch_normalization_layer):
    '''
        Assumes the current layer is Convolution2D layer and the next layer
        is the BatchNormalization layer.
    '''
    gamma, beta, mean, variance = batch_normalization_layer.get_weights()
    epsilon = batch_normalization_layer.epsilon

    # TODO: Add check on weight as bias may not be present.
    weights = convolution2d_layer.get_weights()[0]
    biases = convolution2d_layer.get_weights()[1]

    stdev = np.sqrt(variance + epsilon)
    new_weights = weights * gamma / stdev
    new_biases = (gamma / stdev) * (biases - mean) + beta
    return new_weights, new_biases

def get_merged_layers(current_layer, batch_normalization_layer):
    pass