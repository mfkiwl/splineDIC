"""
.. module:: image_processing
    :platform: Unix, Windows
    :synopsis: Methods for cleaning (sharpen, crop, contrast enhance) and manipulate (warp) images

.. moduleauthor:: Sam Potter <spotter1642@gmail.com>
"""

from . import np
from . import cv2


def im_warp(image, warp_mat):
    """
    Apply an affine warp to an image using OpenCV's affine warp function
    
    :param image: image to warped. Grayscale shape: [N x M]. Color shape: [N x M x 3]
    :type image: ndarray
    :param warp_mat: affine warp matrix. Shape: [2, 3]. See OpenCV documentation for details
    :type warp_mat: ndarray
    :return: warped image
    :rtype ndarray
    """
    
    # Sanitize inputs
    if not(image.ndim == 2 or image.ndim == 3):
        raise ValueError("Image input does not have correct dimensions")
        
    if warp_mat.shape != (2, 3):
        raise ValueError("Warp matrix input does not have correct dimension")
    
    # Get shape of image
    rows, cols = image.shape
    
    # Cast the warp matrix as 32 bit floast
    warp_mat = np.float32(warp_mat)
    
    # Warp the image and require the output to be the same size as input
    warped_im = cv2.warpAffine(image, warp_mat, (cols, rows))
    
    return warped_im
