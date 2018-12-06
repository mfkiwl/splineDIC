"""
.. module:: analysis
    :platform: Unix, Windows
    :synopsis: Module contains functions for analysing displacement, e.g correlation and optimization routines

.. moduleauthor:: Sam Potter <spotter1642@gmail.com>
"""

from . import np
from . import signal
from . import warnings



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



# Could just do this sweeps
def minfun(delta, nodes_ref, ref_im, def_im):
    
    """
    Function for scipy minimizer to minimize

    :param delta: 1D arrary of rigid body rotations
    :type delata: ndarray
    :return: ZNSSD of deformed ref and deformed image
    :rtype: float
    """
    # TODO: Type checks
    
    # Get deltas
    dx = delta[0]
    dy = delta[1]
    
    # Deform the nodes
    # Copy and update reference image locations only
    nodes_def = np.copy(nodes_ref) # Copy
    nodes_def[:, 0] += dx * np.ones(len(nodes_def))
    nodes_def[:, 1] += dy * np.ones(len(nodes_def))
    
    
    # Min/max nodes in x/y for ref and def
    # TODO: Refactor
    # Round values down as the step size must be interger valued
    minx_ref = np.min(nodes_ref[:, 0]).astype('int')
    miny_ref = np.min(nodes_ref[:, 1]).astype('int')
    maxx_ref = np.max(nodes_ref[:, 0]).astype('int')
    maxy_ref = np.max(nodes_ref[:, 1]).astype('int')
    
    minx_def = np.min(nodes_def[:, 0]).astype('int')
    miny_def = np.min(nodes_def[:, 1]).astype('int')
    maxx_def = np.max(nodes_def[:, 0]).astype('int')
    maxy_def = np.max(nodes_def[:, 1]).astype('int')
    
    # Get ref and def images
    ref_subset = ref_im[miny_ref:maxy_ref, minx_ref:maxx_ref]
    def_subset = def_im[miny_def:maxy_def, minx_def:maxx_def]
    
    # Compute ZNSSD
    znssd = discrete_znssd(ref_subset, def_subset)
    
    return znssd
   

def ratchet(maxdx, maxdy, nodes_ref, ref_image, def_image):

	"""
	Compute rigid deformation by ratcheting window over image
	
	:param maxdx: maximum step in x
	:type maxdx: int
	:param maxdy: maximum step in y
	:type maxdy: int
	:param ref_image: reference image array
	:type ref_image: ndarray
`	:param def_image: deformed image array
	:type def_image: ndarray
	:return: displacement [dx dy]
	:rtype:ndarray
	"""
	
	minval = 1000
	
	for i in range(0, maxdx):
		dx = i
		for j in range(0, maxdy):
			dy = j
			delta = np.array([dx, dy])
			retval = minfun(delta, nodes_ref, ref_image, def_image)
			if retval < minval:
				minval = retval
			if np.isclose(minval, 0.0):
				return np.array([dx, dy, minval])
	
	warnings.warn('Could not find exact minimum value')
	return np.array([dx, dy, minval]) 

