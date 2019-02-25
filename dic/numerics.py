"""
.. module:: numerics
    :platform: Unix, Windows
    :synopsis: Module contains functions for basic numerical manipulations of image data, e.g.
    bicubic interpolation, multigrid dimension image resizing, quadrature, masking, etc.

.. moduleauthor:: Sam Potter <spotter1642@gmail.com>
"""

from . import np
from . import signal
from . import sobel
from . import jit


def image_sobel(im_data):

    """
    Compute sobel filter finite differencing on image data and return partial derivatives

    :param im_data: 2D numpy array of data to interpolate. Type should be unsigned int
    :type im_data: ndarray
    :return: tuple of finite difference matrices
    :rtype: tuple
    """

    # Get spatial derivatives via Sobel filter finite differencing
    im_data_x = sobel(im_data, axis=0, mode='constant')
    im_data_y = sobel(im_data, axis=1, mode='constant')
    im_data_xy = sobel(im_data_y, axis=0, mode='constant')

    return im_data_x, im_data_y, im_data_xy


@jit(nopython=True, cache=True)
def bicubic_coefficients(im_data, im_data_x, im_data_y, im_data_xy):

    """
    Compute bicubic interpolation coefficients of an image.

    :param im_data: Image data as 2D array. Type float.
    :type im_data: ndarray
    :param im_data_x: Image data finite difference in x direction. Type float
    :type im_data_x: ndarray
    :param im_data_y: Image data finite difference in y direction. Type float
    :param im_data_xy: Image data finite difference in xy direction. Type float
    :return: Array of bicubic interpolation coefficients. Shape ((rows-1) * (cols-1), 4, 4)
    :rtype: ndarray
    """

    row, col = im_data.shape

    shape = ((row - 1) * (col - 1), 4, 4)

    coeffs = np.zeros(shape, dytpe=np.float64)

    C = np.array([[1., 0., 0., 0.],
                  [0., 0., 1., 0.],
                  [-3., 3., -2., -1.],
                  [2., -2., 1., 1.]])

    D = np.array([[1., 0., -3., 2.],
                  [0., 0., 3., -2.],
                  [0., 1., -2., 1.],
                  [0., 0., -1., 1.]])

    k = 0
    for j in range(0, col - 1):  # Move through x first, which is columns
        for i in range(0, row - 1):  # Move through y next, which is rows
            # Transpose sub-matrices because equation expects top row to be (0, 0), (0, 1) bot row (1, 0), (1, 1)
            F = np.vstack((np.hstack((im_data[i:i + 2, j:j + 2].T, im_data_y[i:i + 2, j:j + 2].T)),
                           np.hstack((im_data_x[i:i + 2, j:j + 2].T, im_data_xy[i:i + 2, j:j + 2].T))))

            A = C @ F @ D

            coeffs[k, :, :] = A

            k += 1

    return coeffs


def image_interp_bicubic(im_data):

    """
    Compute image interpolation coefficients for a single image.

    :param im_data: Image data s 2D array. Type is unsigned int
    :type im_data: ndarray
    :return: Array of bicubic interpolation coefficients. Shape ((rows-1) * (cols-1), 4, 4)
    :rtype: ndarray
    """

    # Get sobel filters
    imx, imy, imxy = image_sobel(im_data)

    coeffs = bicubic_coefficients(im_data.astype(np.float64), imx.astype(np.float64), imy.astype(np.float64),
                                  imxy.astype(np.float64))

    return coeffs

def eval_interp_bicubic(coeffs, x, y, shape):

    """
    Evaluate bicubic interpolation at position x,y on image with corresponding shape and coefficients

    :param coeffs: Array of bicubic coefficients. Shape ((rows - 1) * (cols -1), 4, 4)
    :type coeffs: ndarray
    :param x: X coordinate of interpolation point position
    :type x: float
    :param y: Y coordinate of interpolation point position
    :type y: float
    :param shape: Tuple of interpolation image shape
    :type shape: tuple
    :return: Interpolated value at x, y
    :rtype: float
    """

    row = int(np.floor(y))
    col = int(np.floor(x))

    rows = shape[0] - 1
    cols = shape[1] - 1

    xval = x % 1.0
    yval = y % 1.0

    A = coeffs[col * rows + row, :, :]

    # Switch x and y because of the image coord sys

    xar = np.array([1.0, xval, xval ** 2, xval ** 3])
    yar = np.array([1.0, yval, yval ** 2, yval ** 3])

    p = yar @ A @ xar

    return p


def image_interp_spline(im_data, degree='cubic'):

    """
    Compute either bicubic or biquinitic spline interpolation of 2D image data.

    Bicubic interpolation wraps Scipy's signal processing functionality

    Biquintic interpolation is based on work by
    TODO: CITE UNSER ETC.

    :param im_data: 2D numpy array of data to interpolate. Type can be float or unsigned int
    :type im_data: ndarray
    :param degree: Optional. Flag to control degree of spline interpolation. Default is cubic.
    :type degree: str`
    :return: Matrix of interpolating spline coefficients.
    :rtype: ndarray
    """
    if im_data.ndim != 2:
        raise ValueError('Input data must be a two dimensional array')

    # Compute and return interpolation
    if degree == 'cubic':
        coefficients = signal.cspline2d(im_data)  # Leaving out an additional input options for smoothing for now
    elif degree == 'quintic':
        print('Qunitic functionality not yet implemented')
    else:
        print('Invalid interpolation degree specified! Options are either cubic or quintic')
        return None

    return coefficients


def eval_interp_spline(x, y, image, coeffs=None, order='cubic'):
    """
    Evaluate an image at a non-integer location.

    Allows a user to pass a set of precomputed coeffiecients or let the function perform the interpolation in place

    X is columns
    Y is rows

    :param x: x coordinate of interpolation location
    :type x: float
    :param y: y coordinate of interpolation location
    :type y: float
    :param coeffs: Optional. 2D array of b spline interpolation coefficients
    :type coeffs: ndarray
    :param order: Optional. Order of spline interpolation and evaluation
    :type order: str
    :return: interpolated value as location [x, y]
    :rtype: float
    """

    # Sanitize input
    # Not quite duck typing, but it's very important that the b-spline function recieves floats, not ints as arguments
    if not isinstance(x, float):
        raise TypeError('x coordinate must be a float')

    if not isinstance(y, float):
        raise TypeError('y coordinate must be a float')

    if image.ndim != 2:
        raise ValueError('Image must be 2D array')

    if coeffs.any():
        if not coeffs.shape == image.shape:
            raise ValueError('Coefficient array and Image array must have the same dimension')
    else:
        coeffs = image_interp_spline(image, degree=order)

    # Get the row and column start index based on x, y
    colindex = np.ceil(x - (order + 1) / 2).astype('int')  # Cast to int
    rowindex = np.ceil(y - (order + 1) / 2).astype('int')  # Cast to int

    if order == 'cubic':
        # Alias function
        cubic = signal.cubic

        cols = np.array(range(colindex, colindex + order + 1))
        rows = np.array(range(rowindex, rowindex + order + 1))
        argx = x - cols
        argy = y - rows
        splinex = cubic(argx)
        spliney = cubic(argy)
        val = np.linalg.multi_dot((spliney, coeffs[rowindex: rowindex + order + 1, colindex: colindex + order + 1], splinex))

    else:
        # Alias function
        bspline = signal.bspline

        val = 0.0
        for k in range(rowindex, rowindex + order + 1):  # Adding one to account for range
            for l in range(colindex, colindex + order + 1):
                val += coeffs[k, l] * bspline(y - k, order) * bspline(x - l, order)

    return val
