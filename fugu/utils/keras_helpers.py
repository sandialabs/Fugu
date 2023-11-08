# isort: skip_file
import numpy as np

from scipy.signal import convolve2d

# Turn off black formatting for this file
# fmt: off

# Auxillary/Helper functions
def keras_convolve2d(image,kernel,strides=(1,1),mode="same"):
    '''
        Custom function that uses scipy.signal.convove2d to reproduce the Keras 2D convolution result with strides
    '''
    return np.flip(convolve2d(np.flip(image),np.flip(kernel),mode=mode))[::strides[0],::strides[1]]

def keras_convolve2d_4dinput(image,kernel,strides=(1,1),mode="same",data_format="channels_last",filters=1):
    '''
        Custom function that uses scipy.signal.convove2d to reproduce the Keras 2D convolution result with strides. Attempts to
        handle the number of channels in image and the number of kernels per filter to calculating the result.
    '''
    if data_format.lower() == "channels_last":
        if image.ndim == 3:
            height, width, nChannels = image.shape
        elif image.ndim == 4:
            batch_size, height, width, nChannels = image.shape
    elif data_format.lower() == "channels_first":
        if image.ndim == 3:
            nChannels, height, width = image.shape
        elif image.ndim == 4:
            batch_size, nChannels, height, width = image.shape
    else:
        raise ValueError("Unknown 'data_format' passed to 'keras_convolve2d_4dinput.")

    if data_format.lower() == "channels_last":
        conv2d_answer = np.zeros((height,width,filters))
        for filter in np.arange(filters):
            for channel in np.arange(nChannels):
                id = np.ravel_multi_index((channel,filter),(nChannels,filters))
                print(f"filter: {filter:2d}  channel: {channel:2d}  linearized index: {id:2d}")
                conv2d_answer[:,:,filter] += keras_convolve2d(image[0,:,:,channel],kernel[0,:,:,id],strides,mode) # update zero index in first array position to handle batch_size
    elif data_format.lower() == "channels_first":
        conv2d_answer = np.zeros((filters,height,width))
        for filter in np.arange(filters):
            for channel in np.arange(nChannels):
                id = np.ravel_multi_index((channel,filter),(nChannels,filters))
                conv2d_answer[filter] += keras_convolve2d(image[0,channel,:,:],kernel[0,id,:,:],strides,mode) # update zero index in first array position to handle batch_size

    return conv2d_answer[::strides[0],::strides[1]]

def generate_keras_kernel(nRows,nCols,nFilters,nChannels):
    '''
        Generates an initial kernel (weights) for Keras 2D convolution layer. This essentially, creates an array from an integer sequence (1,2,3,4...) into a format 
        acceptable for the Keras model. The reordering of the columns is so that Keras applies the kernels in (my desired) sequential integer sequence. For instance,
        given two Filters with 2 kernels (channels) per filter given by

        F1 = [[[1,2],[3,4]], [[5,6],[7,8]]]
        F2 = [[[9,10],[11,12]], [[13,14],[15,16]]]

        image = [[[1,2],[3,4]], [[5,6],[7,8]]]

        To respect the filter's two kernels (channels), you must reorder the columns in the Keras kernel so that the first kernel in all filters come first then the second
        kernel in all filters, all the way up to the last kernel in all filters.

        Use the image and filters above, first filter applied to the image should give a 2D convolution result equal to [[184,112],[136,80]]. Similarly, the second filter
        applied to the image gives a 2D convolution result equal to [[472,272],[312,176]].

        Returns a 4d tensor
    '''
    column_permutations = np.concatenate((np.arange(0,nFilters*nChannels,2),np.arange(1,nFilters*nChannels,2)))
    return np.arange(1,nFilters*nChannels*nRows*nCols+1).reshape(nRows*nCols,nFilters*nChannels,order='F')[:,column_permutations].reshape(1,nRows,nCols,nFilters*nChannels)

def generate_mock_image(nRows,nCols,nChannels):
    '''
        Generates a mock image of integers in a sequence. The sequence is from 1 to nRows*nCols*nChannels.

        Returns a 4d tensor
    '''
    return np.arange(1,nRows*nCols*nChannels+1).reshape(nRows*nCols,nChannels,order='F').reshape(1,nRows,nCols,nChannels)