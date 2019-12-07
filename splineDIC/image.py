"""
.. module:: image
    :platform: Unix, Windows
    :synopsis: Module contains class definitions and associated attributes/methods for DIC image objects

.. moduleauthor:: Sam Potter <spotter1642@gmail.com>
"""

from . import os
from . import cv2


class Image:

    """
    Data storage and evaluation class for DIC image data

    Attributes
    data_type: str. String describing type of image data. Options: ['DIC', 'phi', 'DOA']
    image_data: Ndarray. Discrete image data values at each pixel level. Shape Size X x Size Y.

    Methods
    """

    def __init__(self):
        self._image_data = None
        self._data_type = None

    @property
    def data_type(self):
        """
        Image data type
        One of these strings: ['DIC", 'phi', 'DOA']

        :getter: Get image data type
        :setter: Set image data type
        :type: str
        """
        return self._data_type

    @data_type.setter
    def data_type(self, imtype):
        """
        Set image data type

        :param imtype: One of these strings: ['DIC", 'phi', 'DOA']
        :type imtype: str
        :return: None
        :rtype: None
        """
        # Input checking
        if not isinstance(imtype, str):
            raise TypeError('Image data type must be a string')

        accepted_vals = ['DIC', 'phi', 'DOA']

        if imtype not in accepted_vals:
            raise ValueError('Image data type must be one of these: {}'.format(accepted_vals))

        self._data_type = imtype

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

    def load(self, filename):
        """
        Sets image data by loading from an image file

        :param filename: Path to file including extension. Absolute or relative
        :type filename: str
        :return: None
        :rtype: None
        """

        # Input checking
        if not isinstance(filename, str):
            raise TypeError('Filename must be a string')

        if not os.path.isfile(filename):
            raise ValueError('Filename is not a valid path')

        imdata = cv2.imread(filename, -1)  # Read as is

        self.image_data(imdata)
