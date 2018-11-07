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
    # Input checking
    if not isinstance(im_data, np.ndarray):
        raise TypeError('Input data must be a numpy array!')

    if im_data.ndim != 2:
        raise ValueError('Input data must be a two dimensional array')

    # Compute and return interpolation
    if degree == 'cubic':
        coefficients = signal.cspline2d(im_data)  # Leaving out an additional input options for smoothing for now
    elif degree == 'quintic':
        pass
    else:
        print('Invalid interpolation degree specified! Options are either cubic or quintic')
        return None

    return coefficients
