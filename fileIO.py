"""
.. module:: fileIO
    :platform: Unix, Windows
    :synopsis: Methods for reading and writing to files (image data, strain results)

.. moduleauthor:: Sam Potter <spotter1642@gmail.com>
"""

from . import np
from . import sio
from . import os
from . import cv2


def read_mat_file(fname):

    """
    Read in MATLAB .mat file and return the dictionary of values

    :param fname: Name of input file including extension
    :type fname: str
    :return: Dictionary containing .mat file contents
    :rtype: dict
    """
    # Input checking
    if not fname.endswith('.mat'):
        raise ValueError('Input file should have a .mat extension, e.g. my_matlab_data.mat')

    data_dict = sio.loadmat(fname)

    return data_dict


def read_tiff_stack(directory, read_mode=-1):

    """
    Read in all Tiff files in a given directory and return a dictionary of image values.

    :param directory: Full path of directory containing TIFF images.
    :type directory: str
    :param read_mode: Optional. Parameter for controlling OpenCV image read flags. Default is to read as is
    See OpenCV docs for details
    :return: dictionary of the image data. Format {image filename : image data as numpy array}
    :rtype: dict
    """
    # Input checking
    if not isinstance(directory, str):
        raise TypeError('Input must be a full directory path as a string')

    # Save the calling directory
    calling_dir = os.getcwd()

    # Change working directory to input directory
    try:
        os.chdir(directory)
    except OSError as ex:
        print('Operating System Error')
        raise ex

    # Initialize image data dictionary
    image_dict = {}

    # Create list of directory's contents
    directory_contents = os.listdir(directory)

    # List comp over contents and read in only if extension is '.tiff' or '.tif'
    tiff_files = [file for file in directory_contents if file.endswith('.tiff') or file.endswith('.tif')]

    # Loop through the tiff_files and add their data to the dictionary
    for i in range(0, len(tiff_files)):
        image_dict[tiff_files[i]] = cv2.imread(tiff_files[i], read_mode)  # Read file in based on read_mode param

    # Set working directory back to calling directory
    os.chdir(calling_dir)

    return image_dict


def write_gray_tiff_stack(image_array, base_name, directory=None):

    """
    Writes a 3 dimensional array of data to 16 bit gray scale tiff image stacks.
    :param image_array:
    :return:
    """
