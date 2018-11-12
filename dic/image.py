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
    image_data: Ndarray. Discrete image data values at each pixel level. Shape Size X x Size Y.
    data_type: str. String describing type of image data. Options: ['DIC', 'phi', 'DOA']

    Methods
    """

    def __init__(self, image_data, data_type):
        self._image_data = image_data
        self._data_type = data_type

    # Data property definition
    @property
    def image_data(self):
        """
        Image data
        2D Numpy array. Shape Size X x Size Y.

        :getter: Get image data
        :setter: Set image data
        :type: ndarray
        """
        return self._image_data

    # Data property getter
    @image_data.setter
    def image_data(self, array):
        """
        Sets image data and performs type checking

        Provides a consistent way to set the imaging data and directly operates on the class attribute _data
        :param array: input data. Numpy array. Shape Size X x Size Y
        :type array: ndarray
        :return: None
        :rtype: None
        """
        # Input checking: 2D array
        if array.ndim != 2:
            raise ValueError('Data shape must be 2D')

        # Set _data property
        self._image_data = array
