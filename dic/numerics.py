"""
.. module:: numerics
    :platform: Unix, Windows
    :synopsis: Module contains functions for basic numerical manipulations of image data, e.g.
    bicubic interpolation, multigrid dimension image resizing, quadrature, masking, etc.

.. moduleauthor:: Sam Potter <spotter1642@gmail.com>
"""

from . import np
from . import signal


def image_interp(im_data, degree='cubic'):

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


def eval_interp(x, y, image, coeffs=None, order=3):
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
    :type order: int
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
        coeffs = image_interp(image, degree=order)

    # Get the row and column start index based on x, y
    colindex = np.ceil(x - (order + 1) / 2).astype('int')  # Cast to int
    rowindex = np.ceil(y - (order + 1) / 2).astype('int')  # Cast to int

    if order == 3:
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
