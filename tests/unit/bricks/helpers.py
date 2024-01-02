# isort: skip_file
# fmt: off
import numpy as np
from fugu.utils.keras_helpers import keras_convolve2d, keras_convolve2d_4dinput, generate_keras_kernel, generate_mock_image, keras_convolution2d_output_shape, keras_convolution2d_output_shape_4dinput

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
        self.answer_bool = self.answer + self.biases > 0.5
        

    def _get_input_shape_params(self):
        if data_format.lower() == "channels_last":
            batch_size, image_height, image_width, nChannels = self.input_shape
        elif data_format.lower() == "channels_first":
            batch_size, nChannels, image_height, image_width = self.input_shape
        else:
            raise ValueError(f"'data_format' is either 'channels_first' or 'channels_last'. Received {data_format}")
        
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
        batch_size, image_height, image_width, nChannels = get_pool_input_shape_params(input_shape,data_format)
        spatial_input_shape = (image_height, image_width)
        return spatial_input_shape
    
    def _set_convolution_answer(self):
        self.answer = keras_convolve2d_4dinput(self.mock_image, self.filters, self.strides, self.mode, self.data_format, filters=self.nFilters).reshape(self.output_shape)
    
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
        row_stride_positions, col_stride_positions = self.get_stride_positions()
        answer = []

        if self.pool_method == "max":
            self.get_max_pooling_answer(self.pool_input)
        elif self.pool_method == "average": 
            self.get_average_pooling_answer(self.pool_input)
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
        row_stride_positions, col_stride_positions = self.get_stride_positions()
        answer = []

        for row in row_stride_positions[:self.spatial_output_shape[0]]:
            for col in col_stride_positions[:self.spatial_output_shape[1]]:
                for channel in np.arange(self.nChannels):
                    answer.append(np.any(pool_input[0,row:row+self.pool_size[0],col:col+self.pool_size[1],channel]))

        answer = np.reshape(answer, self.output_shape).astype(int)
        answer = (np.array(answer, dtype=int) > 0.9).astype(float)
        return answer
    
    def get_average_pooling_answer(self,pool_input):
        row_stride_positions, col_stride_positions = self.get_stride_positions()
        answer = []

        weights = 1.0 / np.prod(self. pool_size)
        for row in row_stride_positions[:self.spatial_output_shape[0]]:
            for col in col_stride_positions[:self.spatial_output_shape[1]]:
                for channel in np.arange(self.nChannels):
                    answer.append( (weights * pool_input[0,row:row+self.pool_size[0],col:col+self.pool_size[1],channel].astype(int)).sum() )

        answer = np.reshape(answer, self.output_shape).astype(float)
        return answer

    def generate_random_pool_input(self):
        input = np.zeros(self.input_shape,dtype=int).reshape(-1)
        nvals = np.floor(len(input)*0.40).astype(int)
        subset_ids = np.random.choice(np.prod(self.input_shape),nvals,replace=False)
        input[subset_ids] = 1
        return input.reshape(self.input_shape)