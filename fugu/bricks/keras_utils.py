import numpy as np

def input_index_to_matrix_entry(input_shape,basep,bits,index):
    Am, An = input_shape
    linearized_index = np.ravel_multi_index(index,tuple(np.repeat([1,2,basep],[1,2,2])))# zero-based linearized index

    return np.unravel_index(linearized_index,(Am,An,basep*bits))[:2]

def isValueScalar(scalar):
    if not hasattr(scalar, '__len__') and (not isinstance(scalar, str)):
        return True
    else:
        return False

def create_array_from_scalar_value(scalar, shape):
    return scalar * np.ones(shape)

def isShapeCorrect(variable, expected_variable_shape):
    if not type(variable) is np.ndarray:
        variable = np.array(variable)

    if variable.shape == expected_variable_shape:
        return True
    else:
        return False

def is_int_or_2d_tuple_of_ints(variable):
    if isinstance(variable, tuple):
        if len(variable) != 2:
            return False
        else:
            return all([isinstance(val, int) for val in variable])
    elif isinstance(variable, int):
        return True
    else:
        return False

def parse_thresholds_input_parameter(thresholds_input, expected_shape):
    if isValueScalar(thresholds_input):
        return create_array_from_scalar_value(thresholds_input, expected_shape)
    else:
        thresholds = np.array(thresholds_input)
        if isShapeCorrect(thresholds, expected_shape):
            return thresholds
        else:
            raise ValueError(f"Threshold shape {thresholds.shape} does not equal the output neuron shape {expected_shape}.")

def parse_strides_input_parameter(strides_input, error_message):
    if is_int_or_2d_tuple_of_ints(strides_input):
        return strides_input if hasattr(strides_input, "__len__") else tuple(map(int, [strides_input, strides_input]))
    else:
        raise StridesValueError(strides_input, error_message)

class StridesValueError(ValueError):
    def __init__(self, value, message):
        self.value = value
        self.message = message
        super().__init__(message)