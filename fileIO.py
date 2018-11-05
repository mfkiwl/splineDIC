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
from . import plt
from . import cm
from . import utilities


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


def write_tiff_stack(data_array, base_name, directory=None, color_map='viridis'):

    """
    Writes a 3 dimensional array of data to 16 bit tiff image stacks.
    :param data_array: Data array to write to image. Shape N x Image X Size x Image Y Size. Assumed that input is not
    normalized.
    :type data_array: numpy array
    :param base_name: base name string of output file. Will be appended with image index number and .tiff extension
    :type base_name: str
    :param directory: Optional. Full path to write output files into. Default will be current working directory of
    script
    :type directory: str
    :param color_map: Optional. Matplotlib colormap to use for image write.
    :type color_map: str
    :return: None
    :rtype: None
    """
    # Input checking
    if len(data_array.shape) != 3:
        raise ValueError('Data array must be three dimensional!')

    if not isinstance(base_name, str):
        raise TypeError('Base name for image output must be a string')

    # Check if directory needs to be changes
    if directory:
        # Check input
        if not isinstance(directory, str):
            raise ValueError('Output directory path must be a string!')

        # Save current directory for later use
        start = os.getcwd()

        # Change directory
        os.chdir(directory)

    # Loop through every image in array and write output image
    for i in range(len(data_array)):

        # Normalize input data to [0, 1]
        utilities.normalize_2d(data_array[i, :, :])

        # Get colormap
        colormap = cm.get_cmap(color_map)

        # Apply colormap to data
        im_data = colormap(data_array[i, :, :])

        # Scale to 16 bit range
        im_data = 65535 * im_data

        # Convert to unsigned 16 int type
        im_data = np.uint16(im_data)

        # Flip RGB order to BGR order so it's compatible with OpenCV
        im_data = im_data[..., ::-1]

        # Create file output name
        fname = base_name + '_' + str(i) + '.tiff'

        # Write file
        cv2.imwrite(fname, im_data)

    if directory:
        # Move directory back to start
        os.chdir(start)
