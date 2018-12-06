"""
.. module:: analysis
    :platform: Unix, Windows
    :synopsis: Module contains functions for analysing displacement, e.g correlation and optimization routines

.. moduleauthor:: Sam Potter <spotter1642@gmail.com>
"""

from . import np
from . import signal



def discrete_znssd(ref_image, def_image):
    
    """
    Compute the zero normalized sum of square differences (ZNSSD) between two image subsets.
    See Pan et al. Meas Sci Tech 2009 for details
    
    :param ref_image: reference image.
    :type ref_image: ndarray
    :param def_image: deformed image.
    :type def_image: ndarray
    :return: ZNNSD
    :rtype: float
    """
    
    # Sanitize imputs
    if ref_image.ndim != 2:
        raise ValueError('Reference image input must be 2d')
        
    if def_image.ndim != 2:
        raise ValueError('Deformed image input must be 2d')
    
    if ref_image.shape != def_image.shape:
        raise ValueError('Shape of reference and deformed images must match')
    
    # Compute mean of reference image
    fm = np.mean(ref_image)
    
    # compute mean of deformed image
    gm = np.mean(def_image)
    
    # Compute std dev of reference image
    fstd = np.std(ref_image)
    
    # Compute std dev of deformed image
    gstd = np.std(def_image)
    
    # Compute ZNSSD
    
    znssd = 0.0
    
    for i in range(0, ref_image.shape[0]):
        for j in range(0, ref_image.shape[1]):
            znssd += ((ref_image[i, j] - fm) / fstd - (def_image[i, j] - gm) / gstd)**2
    
    return znssd


