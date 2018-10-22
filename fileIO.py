"""
.. module:: fileIO
    :platform: Unix, Windows
    :synopsis: Methods for reading and writing to files (image data, strain results)

.. moduleauthor:: Sam Potter <spotter1642@gmail.com>
"""

from . import np
from . import sio


def read_mat(fname):

    """
    Read in MATLAB .mat file and return the dictionary of values

    :param fname: Name of input file including extension
    :type fname: str
    :return: Dictionary containing .mat file contents
    :rtype: dict
    """

    data_dict = sio.loadmat(fname)

    return data_dict
