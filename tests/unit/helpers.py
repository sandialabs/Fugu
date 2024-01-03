# isort: skip_file
# fmt: off
import numpy as np
from fugu.utils.keras_helpers import keras_convolve2d, keras_convolve2d_4dinput, generate_keras_kernel, generate_mock_image, keras_convolution2d_output_shape, keras_convolution2d_output_shape_4dinput

from tensorflow.keras.models import Sequential
from tensorflow.keras import Model, initializers
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D

class ConvolutionParams:

    def __init__(self, image_height=3, image_width=3, nChannels=2, kernel_height=2, kernel_width=2, nFilters=3, strides=(1,1), mode="same", data_format="channels_last", batch_size=1, biases=None):
        self.image_height = image_height
        self.image_width = image_width
        self.nChannels = nChannels
        self.kernel_height = kernel_height
        self.kernel_width = kernel_width
        self.nFilters = nFilters
        self.strides = strides
        self.mode = mode
        self.thresholds = 0.5
        self.data_format = data_format
        self.batch_size = batch_size

        if biases is None:
            self.biases = np.zeros((self.nFilters,))
        elif len(biases) != self.nFilters:
            raise ValueError(f"Number of biases does not match the number of filters.")
        else:
            self.biases = biases

        self.filters = generate_keras_kernel(self.kernel_height,self.kernel_width,self.nFilters,self.nChannels)
        self.spatial_input_shape = (image_height,image_width)
        self.mock_image = generate_mock_image(self.image_height,self.image_width,self.nChannels)
        self._set_input_shape()
        self._set_output_shape()
        self._set_convolution_answer()
        self._set_convolution_answer_boolean()

    def _get_input_shape_params(self):
        if self.data_format.lower() == "channels_last":
            batch_size, image_height, image_width, nChannels = self.input_shape
        elif self.data_format.lower() == "channels_first":
            batch_size, nChannels, image_height, image_width = self.input_shape
        else:
            raise ValueError(f"'data_format' is either 'channels_first' or 'channels_last'. Received {self.data_format}")
        
        return batch_size, image_height, image_width, nChannels

    def _set_input_shape(self):
        if self.data_format == "channels_last":
            self.input_shape = (self.batch_size,self.image_height,self.image_width,self.nChannels)
        elif self.data_format == "channels_first":
            self.input_shape = (self.batch_size,self.nChannels,self.image_height,self.image_width)
        else:
            raise ValueError(f"'data_format' is either 'channels_first' or 'channels_last'. Received {self.data_format}")

    def _set_output_shape(self):
        strides_shape = np.array(self.strides)
        input_shape = np.array(self.spatial_input_shape)
        kernel_shape = np.array(self.filters.shape)[:2]
        nFilters = self.nFilters

        p = 0.5 if self.mode == "same" else 0
        output_shape = np.floor((input_shape + 2*p - kernel_shape)/strides_shape + 1).astype(int)

        if self.data_format == "channels_last":
            self.output_shape = (self.batch_size,) + tuple(output_shape) + (nFilters,)
        elif self.data_format == "channels_first":
            self.output_shape = (self.batch_size,) + (nFilters,) + tuple(output_shape)
        else:
            raise ValueError(f"'data_format' is either 'channels_first' or 'channels_last'. Received {self.data_format}")
        
    def _get_spatial_input_shape(self):
        batch_size, image_height, image_width, nChannels = self._get_input_shape_params()
        spatial_input_shape = (image_height, image_width)
        return spatial_input_shape
    
    def _set_convolution_answer(self):
        self.answer = keras_convolve2d_4dinput(self.mock_image, self.filters, self.strides, self.mode, self.data_format, filters=self.nFilters).reshape(self.output_shape)

    def _set_convolution_answer_boolean(self):
        if not hasattr(self,"answer"):
            self._set_convolution_answer()

        self.answer_bool = self.answer + self.biases > 0.5

    def set_convolution_answer_boolean(self, biases):
        if not hasattr(self,"answer"):
            self._set_convolution_answer()

        self.answer_bool = self.answer + biases > 0.5

    def get_random_biases_within_answer_range(self, nSamples=1):
        mins = np.min(self.answer, axis=(0,1,2))
        maxs = np.max(self.answer, axis=(0,1,2))
        if nSamples == 1:
            size = (self.nFilters,)
        else:
            size = (nSamples,self.nFilters)
        return -np.random.randint(mins,maxs,size=size).astype(float)

    def calculate_convolution_answer_boolean(self, biases):
        if not hasattr(self,"answer"):
            self._set_convolution_answer()

        return self.answer + biases > 0.5

    def calculate_convolution(self, image):
        return keras_convolve2d_4dinput(image, self.filters, self.strides, self.mode, self.data_format, filters=self.nFilters).reshape(self.output_shape)
    

class PoolingParams:
    
    def __init__(self,params_convo, pool_size=(2,2), pool_strides=(1,1), pool_padding="same", pool_method="max"):
        self._set_and_validate_pool_size_input(pool_size)
        self._set_and_validate_stride_input(pool_strides)
        self.pool_height, self.pool_width = pool_size
        self.pool_size = pool_size
        self.pool_padding = pool_padding
        self.pool_method = pool_method
        self.pool_thresholds = 0.9 #TODO: Update for "average" method; current value is for "max" only.
        self.pool_input = params_convo.answer_bool.astype(int)

        self.data_format = params_convo.data_format
        self.input_shape = params_convo.output_shape
        self.nChannels = self._get_pool_input_shape_params()[3]
        self._set_spatial_input_shape()
        self._set_spatial_output_shape()
        self._set_spatial_window_shape()
        self._set_output_shape()
        self._set_pooling_answer()

    def _set_and_validate_pool_size_input(self,pool_size):
        if hasattr(pool_size,"__len__"):
            if type(pool_size) is not tuple:
                raise ValueError("'pool_size' must be an integer or tuple of 2 integers.")

            if len(pool_size) != 2:
                raise ValueError("'pool_size' must be an integer or tuple of 2 integers.")
        elif isinstance(pool_size,float):
            raise ValueError("'pool_size' must be an integer or tuple of 2 integers.")
        else:
            pool_size = (pool_size, pool_size)

        self.pool_size = pool_size

    def _set_and_validate_stride_input(self,strides):
        if strides is None:
            strides = self.pool_size
        elif hasattr(strides, "__len__"):
            if type(strides) is not tuple:
                raise ValueError("'strides' must be an integer, tuple of 2 integers, or None. If None then defaults to 'pool_size'")

            if len(strides) != 2:
                raise ValueError("'strides' must be an integer, tuple of 2 integers, or None. If None then defaults to 'pool_size'")
        elif isinstance(strides, float):
            raise ValueError("'strides' must be an integer, tuple of 2 integers, or None. If None then defaults to 'pool_size'")
        else:
            strides = (strides, strides)
        self.pool_strides = strides

    def _set_output_shape(self):
        if self.data_format.lower() == "channels_last":
            output_shape = (self.batch_size, *self.spatial_output_shape, self.nChannels)
        else:
            output_shape = (self.batch_size, self.nChannels, *self.spatial_output_shape)

        self.output_shape = output_shape
    
    def _get_spatial_output_shape(self):
        if self.pool_padding.lower() == "same":
            spatial_output_shape = self._same_padding_spatial_output_shape()
        elif self.pool_padding.lower() == "valid":
            spatial_output_shape = self._valid_padding_spatial_output_shape()
        else:
            raise ValueError(f"'pool_padding' is one of 'same' or 'valid'. Received {self.pool_padding}.")

        spatial_output_shape = tuple(map(int,spatial_output_shape))
        return spatial_output_shape

    def _set_spatial_output_shape(self):
        if self.pool_padding.lower() == "same":
            spatial_output_shape = self._same_padding_spatial_output_shape()
        elif self.pool_padding.lower() == "valid":
            spatial_output_shape = self._valid_padding_spatial_output_shape()
        else:
            raise ValueError(f"'pool_padding' is one of 'same' or 'valid'. Received {self.pool_padding}.")

        self.spatial_output_shape = tuple(map(int,spatial_output_shape))
            
    def _get_spatial_input_shape(self):
        self.batch_size, self.image_height, self.image_width, self.nChannels = self._get_pool_input_shape_params()
        spatial_input_shape = (self.image_height, self.image_width)
        return spatial_input_shape

    def _set_spatial_input_shape(self):
        self.batch_size, self.image_height, self.image_width, self.nChannels = self._get_pool_input_shape_params()
        self.spatial_input_shape = (self.image_height, self.image_width)

    def _get_spatial_window_shape(self):
        spatial_window_shape = (self.stride_positions(self.spatial_input_shape[0],self.pool_strides[0])[-1] + self.pool_size[0], 
                                self.stride_positions(self.spatial_input_shape[1],self.pool_strides[1])[-1] + self.pool_size[1])
        return spatial_window_shape

    def _set_spatial_window_shape(self):
        self.spatial_window_shape = (self.stride_positions(self.spatial_input_shape[0],self.pool_strides[0])[-1] + self.pool_size[0], 
                                     self.stride_positions(self.spatial_input_shape[1],self.pool_strides[1])[-1] + self.pool_size[1])

    def _same_padding_spatial_output_shape(self):
        if not hasattr(self,"spatial_input_shape"):
            self.spatial_input_shape = self._get_spatial_input_shape()

        return np.floor((np.array(self.spatial_input_shape) - 1) / np.array(self.pool_strides)) + 1

    def _valid_padding_spatial_output_shape(self):
        if not hasattr(self,"spatial_input_shape"):
            self.spatial_input_shape = self._get_spatial_input_shape()

        return np.floor((np.array(self.spatial_input_shape) - np.array(self.pool_size)) / np.array(self.pool_strides)) + 1
    
    def _get_pool_input_shape_params(self):
        if self.data_format.lower() == "channels_last":
            batch_size, image_height, image_width, nChannels = self.input_shape
        elif self.data_format.lower() == "channels_first":
            batch_size, nChannels, image_height, image_width = self.input_shape
        else:
            raise ValueError(f"'data_format' is either 'channels_first' or 'channels_last'. Received {self.data_format}")

        return batch_size, image_height, image_width, nChannels
    
    def stride_positions(self,pixel_dim, stride_len):
        return np.arange(0, pixel_dim, stride_len, dtype=int)

    def get_stride_positions(self):
        return [self.stride_positions(self.spatial_input_shape[0],self.pool_strides[0]), self.stride_positions(self.spatial_input_shape[1],self.pool_strides[1])]
    
    def _set_pooling_answer(self):
        answer = []

        if self.pool_method == "max":
            answer = self.get_max_pooling_answer(self.pool_input)
        elif self.pool_method == "average": 
            answer = self.get_average_pooling_answer(self.pool_input)
        else:
            print(f"'method' class member variable must be either 'max' or 'average'. But it is {self.pool_method}.")
            raise ValueError("Unrecognized 'method' class member variable.")
            
        self.pool_answer = answer
    
    
    def calculate_pooling_answer(self, pool_input):
        if self.pool_method == "max":
            answer = self.get_max_pooling_answer(pool_input)
        elif self.pool_method == "average": 
            answer = self.get_average_pooling_answer(pool_input)
        else:
            print(f"'method' class member variable must be either 'max' or 'average'. But it is {self.pool_method}.")
            raise ValueError("Unrecognized 'method' class member variable.")
            
        return answer
    
    def get_max_pooling_answer(self, pool_input):
        padded_input_shape_bounds = self.get_padded_input_shape_bounds()
        row_stride_positions, col_stride_positions = self.get_stride_positions_from_bounds(padded_input_shape_bounds)
        answer = []

        for row in row_stride_positions[:self.spatial_output_shape[0]]:
            irow, frow = self.adjust_position_to_input_length(row,self.spatial_input_shape[0],self.pool_size[0])
            for col in col_stride_positions[:self.spatial_output_shape[1]]:
                icol, fcol = self.adjust_position_to_input_length(col,self.spatial_input_shape[1],self.pool_size[1])
                for channel in np.arange(self.nChannels):
                    answer.append(np.any(pool_input[0,irow:frow,icol:fcol,channel]))

        answer = np.reshape(answer, self.output_shape).astype(int)
        answer = (np.array(answer, dtype=int) > 0.9).astype(float)
        return answer
    
    def get_average_pooling_answer(self,pool_input):
        padded_input_shape_bounds = self.get_padded_input_shape_bounds()
        row_stride_positions, col_stride_positions = self.get_stride_positions_from_bounds(padded_input_shape_bounds)
        answer = []

        weights = 1.0 / np.prod(self. pool_size)
        for row in row_stride_positions[:self.spatial_output_shape[0]]:
            irow, frow = self.adjust_position_to_input_length(row,self.spatial_input_shape[0],self.pool_size[0])
            for col in col_stride_positions[:self.spatial_output_shape[1]]:
                icol, fcol = self.adjust_position_to_input_length(col,self.spatial_input_shape[1],self.pool_size[1])
                for channel in np.arange(self.nChannels):
                    answer.append( (weights * pool_input[0,irow:frow,icol:fcol,channel].astype(int)).sum() )

        answer = np.reshape(answer, self.output_shape).astype(float)
        return answer

    def generate_random_pool_input(self):
        input = np.zeros(self.input_shape,dtype=int).reshape(-1)
        nvals = np.floor(len(input)*0.40).astype(int)
        subset_ids = np.random.choice(np.prod(self.input_shape),nvals,replace=False)
        input[subset_ids] = 1
        return input.reshape(self.input_shape)

    def adjust_position_to_input_length(self, pos, input_length, pool_length):
        if pos < 0:
            ipos = 0
            fpos = pos + pool_length
        elif pos + pool_length > input_length:
            ipos = input_length - pool_length + 1
            fpos = input_length
        else:
            ipos = pos
            fpos = pos + pool_length

        return ipos, fpos

    def get_stride_positions_for_window(self):
        lb, ub = self.adjust_row_col_for_window_shape()
        row_positions = np.arange(lb[0],ub[0],self.pool_strides[0])
        col_positions = np.arange(lb[1],ub[1],self.pool_strides[1])
        return row_positions, col_positions

    def adjust_row_col_for_window_shape(self):
        input_shape = np.array(self.spatial_input_shape)
        window_shape = np.array(self.spatial_window_shape)
        lb = np.ceil(0.5*(input_shape - window_shape)).astype(int)
        ub = np.ceil(0.5*(input_shape + window_shape)).astype(int)
        return lb, ub

    def get_stride_positions_from_bounds(self, input_shape_bounds):
        return [np.arange(input_shape_bounds[0,0],input_shape_bounds[0,1],self.pool_strides[0]),
                np.arange(input_shape_bounds[1,0],input_shape_bounds[1,1],self.pool_strides[1])]

    def get_padding_shape(self):
        padding_shape = (self.pool_size[0] - 1, self.pool_size[1] - 1)
        return padding_shape

    def get_padded_input_shape_bounds(self):
        '''
            padding_shape = (prow,pcol)

            If the kernel height is odd then we will pad prow/2 rows on top and bottom. However, if the kernel height 
            is even then we pad floor(prow/2) rows on the top and ceil(prow/2) rows on the bottom. The width is padded
            in a similar fashion.
        '''
        padding_shape = self.get_padded_zeros_count()
        assert self.check_padded_zeros_count(padding_shape)

        # determine padding adjustments for top/bottom/left/right regions.
        top_bottom_padding = self.get_padding_amount(padding_shape[0])
        left_right_padding = self.get_padding_amount(padding_shape[1])

        # add adjustments to input_shape and return result
        top_bottom_bounds = [0 - top_bottom_padding[0], top_bottom_padding[1] + self.spatial_input_shape[1]]
        left_right_bounds = [0 - left_right_padding[0], left_right_padding[1] + self.spatial_input_shape[1]]
        return np.array([top_bottom_bounds, left_right_bounds]).astype(int)

    def get_padding_amount(self, dim_length):
        adjustments_array = np.array([np.floor(0.5*dim_length), np.ceil(0.5*dim_length)])
        return adjustments_array

    def get_padded_zeros_count(self):
        '''
            p = s*(g-1) + k - n
        '''
        n = np.array(self.spatial_input_shape)
        k = np.array(self.pool_size)
        s = np.array(self.pool_strides)
        g = np.array(self.spatial_output_shape)

        padding_count = (s*(g - 1) + k - n)
        padding_count[padding_count < 0] = 0.
        return tuple(padding_count.astype(int))

    def check_padded_zeros_count(self, padding_count):
        n = np.array(self.spatial_input_shape)
        k = np.array(self.pool_size)
        s = np.array(self.pool_strides)
        g = np.array(self.spatial_output_shape)
        p = np.array(padding_count)
        calculated_output_shape = np.floor((n-k+p+s)/s)
        expected_output_shape = g
        isSameOutputShape = expected_output_shape == calculated_output_shape
        return isSameOutputShape.all()

class DenseParams:
    pass

class KerasParams:
    def __init__(self,params_layers_list):
        model = Sequential()
        for layerID, params_layer in enumerate(params_layers_list):

            if layerID == 0:
                if params_layer.data_format == "channels_last":
                    input_shape = tuple(list(params_layer.input_shape)[1:])
                elif params_layer.data_format == "channels_first":
                    input_shape = None

            if isinstance(params_layer, ConvolutionParams):
                model = self.add_convolution_layer(model,params_layer,layerID,input_shape)

            if isinstance(params_layer, PoolingParams):
                model = self.add_pooling_layer(model, params_layer, layerID, input_shape)

            if isinstance(params_layer, DenseParams):
                if layerID == 0:
                    pass
                else:
                    pass

        feature_extractor = Model(inputs=model.inputs, outputs=[layer.output for layer in model.layers])
        self.model = model
        self.features_extractor = feature_extractor

    def add_convolution_layer(self, model, params_layer, layerID, input_shape=None):
        if input_shape is None:
            model.add(Conv2D(params_layer.nFilters,
                            (params_layer.kernel_height, params_layer.kernel_width),
                            strides=params_layer.strides,
                            padding=params_layer.mode,
                            activation=None,
                            use_bias=True,
                            name=str(layerID),
                            kernel_initializer=initializers.constant(np.flip(params_layer.filters,(0,1))),
                            bias_initializer=initializers.constant(np.array(params_layer.biases).reshape((params_layer.nFilters,)))))
        else:
            model.add(Conv2D(params_layer.nFilters,
                            (params_layer.kernel_height, params_layer.kernel_width),
                            strides=params_layer.strides,
                            padding=params_layer.mode,
                            activation=None,
                            use_bias=True,
                            input_shape=input_shape,
                            name=str(layerID),
                            kernel_initializer=initializers.constant(np.flip(params_layer.filters,(0,1))),
                            bias_initializer=initializers.constant(params_layer.biases)))
        return model

    def add_pooling_layer(self, model, params_layer, layerID, input_shape=None):
        if input_shape is None:
            model.add(MaxPooling2D(pool_size=params_layer.pool_size, strides=params_layer.pool_strides, padding=params_layer.pool_padding, name=str(layerID)))
        else:
            model.add(MaxPooling2D(pool_size=params_layer.pool_size, strides=params_layer.pool_strides, padding=params_layer.pool_padding, name=str(layerID), input_shape=input_shape))
        return model

    def add_dense_layer(self, model, params_layer, layerID, input_shape=None):
        return model