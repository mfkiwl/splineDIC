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
from . import cm
from . import utilities
from . import errno


def read_mat_file(fname):

    """
    Read in MATLAB .mat file and return the dictionary of values

    :param fname: Name of input file including extension
    :type fname: str
    :return: Dictionary containing .mat file contents
    :rtype: dict
    """
    # Input checking
    # This check does not exactly conform to "Duck Typing", but I think it's useful because the exception thrown by
    # the Scipy IO module is a little vague and can be confusing.
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
    # Save the calling directory
    calling_dir = os.getcwd()

    # Change working directory to input directory
    try:
        os.chdir(directory)
    except OSError as ex:
        print('Operating System Error:')
        print(OSError.strerror)
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

    Allows options for specifying input directory and color map
    :param data_array: Data array to write to image. Shape Number of Image Steps x Image X Size x Image Y Size.
    Assumed that input is not normalized.
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

    # Check if directory needs to be changes
    if directory:
        # Save current directory for later use
        start = os.getcwd()

        # Change directory
        # Attempt to move into directory
        try:
            os.chdir(directory)
        except OSError as ex:
            if ex.errno != errno.ENOENT:
                raise
        # If directory doesn't exist, try to make it
        try:
            os.makedirs(directory)
            os.chdir(directory)
        except OSError:
            print('Unable to move into or create requested output directory')
            print(directory)
            raise

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


def write_surface_to_txt(surface_instance, filename):
    """
    Save Spline Surface parametrization data to text file.

    Saves degree, knot vector, and control points.

    :param surface_instance: Surface object to save
    :type surface_instance: spline.Surface() object
    :param filename: Path (may be relative or absolute) to destination file. Must include extension.
    :type filename: str
    :return: None
    :rtype: None
    """
    with open(filename, 'w') as f:
        # Write degree information
        f.write('Degree - p:\n')
        f.write('{}\n'.format(surface_instance.degree_u))

        f.write('Degree - q:\n')
        f.write('{}\n'.format(surface_instance.degree_v))

        # Write knot vector information
        f.write('Number of Knots in U\n')
        f.write('{}\n'.format(len(surface_instance.knot_vector_u)))

        f.write('Number of Knots in V\n')
        f.write('{}\n'.format(len(surface_instance.knot_vector_v)))

        f.write('Knot Vector - U\n')
        for knot in surface_instance.knot_vector_u:
            f.write('{}\t'.format(knot))
        f.write('\n')

        f.write('Knot Vector - V\n')
        for knot in surface_instance.knot_vector_v:
            f.write('{}\t'.format(knot))
        f.write('\n')

        # Write control points info
        f.write('Number of Control Points in U\n')
        f.write('{}\n'.format(surface_instance.num_ctrlpts_u))

        f.write('Number of Control Points in V\n')
        f.write('{}\n'.format(surface_instance.num_ctrlpts_v))

        f.write('Control Points\n')
        for ctrlpt in surface_instance.control_points:
            f.write('{}\t {}\t {}\n'.format(ctrlpt[0], ctrlpt[1], ctrlpt[2]))


def read_surf_from_txt(surface_instance, filename):
    """
    Read spline surface data from file and modify an instance of the spline.Surface() class

    :param surface_instance: Surface object to define with data in filename
    :type surface_instance: spline.Surface() object
    :param filename: Path (relative or absolute) to file containing curve data. Must include extension.
    :type filename: str
    :return: None
    :rtype: None
    """

    with open(filename, 'r') as f:
        contents = [line.strip('\n') for line in f]

        # Pull degree
        degree_u = int(contents[1])
        degree_v = int(contents[3])

        # Get number of knots
        num_knots_u = int(contents[5])
        num_knots_v = int(contents[7])

        # Get knot values
        knot_vector_u = np.array(list(map(float, contents[9].split('\t'))))
        knot_vector_v = np.array(list(map(float, contents[11].split('\t'))))

        # Get number of control points
        num_ctrlpts_u = int(contents[13])
        num_ctrlpts_v = int(contents[15])

        # Get actual control points
        control_points = [list(map(float, contents[line].split())) for line in range(17, len(contents))]
        control_points = np.array(control_points)

        # Setup the curve
        surface_instance.degree_u = degree_u
        surface_instance.degree_v = degree_v

        surface_instance.num_ctrlpts_u = num_ctrlpts_u
        surface_instance.num_ctrlpts_v = num_ctrlpts_v

        surface_instance.control_points = control_points

        surface_instance.knot_vector_u = knot_vector_u
        surface_instance.knot_vector_v = knot_vector_v
