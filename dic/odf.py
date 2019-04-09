"""
.. module:: odf
    :platform: Unix, Windows
    :synopsis: Module contains functions for computing and manipulating orientation distribution data

.. moduleauthor:: Sam Potter <spotter1642@gmail.com>
"""

from . import np


def Ifiber(a0, a2, a4, phi, theta):

    '''
    Compute the double cosine series solution to Mie scattering of cylindrical fibers.

    :param a0: a0 parameter
    :type a0: float
    :param a2: a2 parameter
    :type a2: float
    :param a4: a4 parameter
    :type a4: float
    :param phi: preferred fiber direction in degrees
    :type phi: float
    :param theta:
    :return:
    '''

    vals = a0 + a2 * np.cos(np.deg2rad(2 * (theta - phi))) + a4 * np.cos(np.deg2rad(4 * (theta - phi)))

    return vals
