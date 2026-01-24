import numpy as np

from fugu.utils.types import bool_types, float_types, str_types
from fugu.utils.validation import int_to_float, validate_type


class InputEncoding:
    """
    Base class for input encoding schemes that convert input data into a format suitable for input neurons.
    This class provides a method to get an iterable representation of the input data based on the specified encoding scheme.

    """

    def __init__(self, encoding_scheme: str = None, in_stream=None, frequency: int = None, bins: int = None):
        self.encoding = encoding_scheme
        self.in_stream = in_stream
        self.fr = frequency
        self.bins = bins

    def get_iterable(self):

        if self.encoding == "Poisson":
            # Convert to poisson iterable if the input stream is an array
            if type(self.in_stream) is np.ndarray:
                self.in_stream = self.in_stream.flatten()

            # If the data is already a stream, pass it as is
            if len(self.in_stream) > 1:
                return self.in_stream
            else:
                # If the data is a singular value, convert it into a poisson stream iterable of length equalling bin sizes
                dt = 0.001
                fr2 = self.fr * self.in_stream[0]
                poisson_output = np.random.rand(1, self.bins) < fr2 * dt
                poisson_output = poisson_output.astype(int)
                return poisson_output[0]

        if self.encoding == "Binary":
            # Convert to binary iterable if the input stream is an array
            if type(self.in_stream) is np.ndarray:
                self.in_stream = self.in_stream.flatten()

            # If the data is already a stream, pass it as is
            if len(self.in_stream) > 1:
                return self.in_stream
            else:
                # If the data is a singular value, convert it into a binary stream iterable of length equalling bin sizes
                binary_output = np.zeros(self.bins)
                binary_output[int(self.in_stream[0])] = 1
                return binary_output
