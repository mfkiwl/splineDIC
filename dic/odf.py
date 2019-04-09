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
    :param theta: Values of theta at which to evaluate the cosine series. Values in degrees
    :type theta: ndarray
    :return: Intensity values
    :rtype: ndarray
    '''

    vals = a0 + a2 * np.cos(np.deg2rad(2 * (theta - phi))) + a4 * np.cos(np.deg2rad(4 * (theta - phi)))

    return vals


def syntheticIdist(a0, a2, a4, phi, theta, splay, nsamples, distribution='uniform'):

    '''
    Function for generating a synthetic pSFDI signal from a distribution of fibers directions about a mean of phi.

    :param a0:
    :param a2:
    :param a4:
    :param phi: preferred fiber direction in degrees
    :type phi: float
    :param theta: Values of theta at which to evaluate the cosine series. Values in degrees
    :type theta: ndarray
    :param splay: standard deviation of fiber direction distribution
    :type splay: float
    :param nsamples: number of samples to draw from the statistical distribution
    :type nsamples: int
    :param distribution: Optional. Specify type of distribution to draw samples from. Default is uniform
    :type distribution: str
    :return: Return a periodic signal representing the summed contributions of each single fiber sample to the over all
    signal
    :rtype: ndarray
    '''

    Idist = np.zeros((nsamples, len(theta)))
    for i in range(0, nsamples):
        if distribution == 'uniform':
            phi = np.random.uniform(-1 * splay, splay)
        elif distribution == 'normal':
            phi = np.random.normal(0, splay)

        vals = Ifiber(a0, a2, a4, phi, theta)
        Idist[i, :] = vals

    Idist = np.sum(Idist, axis=0) / nsamples

    return Idist