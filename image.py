"""
.. module:: image
    :platform: Unix, Windows
    :synopsis: Module contains class definitions and associated attributes/methods for DIC image objects

.. moduleauthor:: Sam Potter <spotter1642@gmail.com>
"""

from . import np


class Image:

    """
    Data storage and evaluation class for DIC image data

    Attributes
    data: discrete image data values at each pixel level. Shape Size X x Size Y.

    Methods
    """

    def __init__(self, data):
        self._data = data

    # Data property definition
    @property
    def data(self):
        """
        Image data
        2D Numpy array. Shape Size X x Size Y.

        :getter: Get image data
        :setter: Set image data
        :type: ndarray
        """
        return self._data

    # Data property getter
    @data.setter
    def data(self, array):
        """
        Sets image data and performs type checking

        Provides a consistent way to set the imaging data and directly operates on the class attribute _data
        :param array: input data. Numpy array. Shape Size X x Size Y
        :type array: ndarray
        :return: None
        :rtype: None
        """
        # Input checking: Numpy array
        if not isinstance(array, np.ndarray):
            raise TypeError('Data must be a 2D Numpy Array')

        # Input checking: 2D array
        if array.ndim != 2:
            raise ValueError('Data shape must be 2D')

        # Set _data property
        self._data = array
