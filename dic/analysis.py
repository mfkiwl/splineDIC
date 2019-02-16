"""
.. module:: analysis
    :platform: Unix, Windows
    :synopsis: Module contains functions for analysing displacement, e.g correlation and optimization routines

.. moduleauthor:: Sam Potter <spotter1642@gmail.com>
"""

from . import np
from . import warnings
from . import numerics
from . import geomdl
from geomdl import BSpline as bs
from geomdl import utilities as gutil


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
            znssd += ((ref_image[i, j] - fm) / fstd - (def_image[i, j] - gm) / gstd) ** 2

    return znssd


def mesh(ref_cpts, degree=3):
    """
    Generate Bspline analysis mesh from set of control points

    :param ref_cpts: N x 2 list of reference mesh control point positions
    :type ref_cpts: ndarray
    :param degree: Optional. Degree of B-spline curve
    :typ degree: int
    :return: B-spline surface with a uniform knot vector and specified control points and (optionally) degree
    :rtype: Bspline surface object
    """

    # Sanitize input
    if ref_cpts.ndim != 2:
        raise ValueError('Control points must have shape N x 2, with each row as [X Y]')

    # Get number of control points
    num_ctrlpts = np.sqrt(len(ref_cpts)).astype('int')

    # Initialize surface
    surf = bs.Surface()

    # Set degree
    surf.degree_u = degree
    surf.degree_v = degree

    # Set control points
    surf.set_ctrlpts(ref_cpts, num_ctrlpts, num_ctrlpts)

    # Set knots
    surf.knotvector_u = gutil.generate_knot_vector(surf.degree_u, num_ctrlpts)
    surf.knotvector_v = gutil.generate_knot_vector(surf.degree_v, num_ctrlpts)

    surf.delta = 0.001

    return surf



# TODO: Rewrite this and just use projection and Newton's methods
def parameterize_pixels(ref_image, ref_mesh):
    """
    For each pixel in the reference image region of interest, compute the nearest u,v coordinate pair in the
    spline mesh

    :param ref_image: reference image
    :type ref_image: ndarray
    :param ref_mesh: Bspline surface mesh of on reference image
    :type ref_mesh: Bspine surface object
    :return: Array of u, v parametric coordinates saved at pixel coordinate [Pixel u, Pixel v]
    :rtype: ndarray
    """

    # Sanitize input

    # Define a 3D array to contain u, v values
    uv_vals = np.zeros((2,) + ref_image.shape)

    # Get mesh control points
    ref_cpts = np.array(ref_mesh.ctrlpts)

    # Get min and max column values from min/max reference ctrlpt node x values
    colmin = np.min(ref_cpts[:, 0]).astype('int')
    colmax = np.max(ref_cpts[:, 0]).astype('int')

    # Get min and max row values from min/max reference ctrlpt node y values
    rowmin = np.min(ref_cpts[:, 1]).astype('int')
    rowmax = np.max(ref_cpts[:, 1]).astype('int')

    # Precompute the mesh surface points at u,v values
    mesh_pts = np.array(ref_mesh.evalpts)

    # Get the eval delta from the surf
    delta = ref_mesh.delta[0]

    # Get divisor
    divisor = 1 / delta

    # Loop through pixels
    for i in range(rowmin, rowmax):
        for j in range(colmin, colmax):
            # Get pixel coordinate value
            val = [j, i]  # [x, y]

            # Compute Euclidean distance between pixel coordinat and all computed ref mesh surf pts
            diff = np.sqrt(np.square(mesh_pts[:, 0] - val[0]) + np.square(mesh_pts[:, 1] - val[1]))

            # Get index value of minimum distance
            idx = np.where(diff == diff.min())[0]  # where returns a tuple, unpack
            idx = idx[0]  # Grab actual integer value from array

            # Get u value from divisor
            u = delta * (idx // divisor)

            # Get v value from remainder
            v = delta * (idx % divisor)

            # Add u,v values to array
            uv_vals[0, i, j] = u
            uv_vals[1, i, j] = v

    return uv_vals


def deform_mesh(ref_mesh, cpts_disp):
    """
    Produce a new B-spline surface mesh by adding a displacement to each control point of a reference mesh
    :param ref_mesh: reference state B-spline mesh
    :type ref_mesh: Bspline surface
    :param cpts_disp: control point displacements [delta X, delta Y]
    :type cpts_disp: ndarray
    :return: deformed B-spline surface mesh
    :rtype: Bspline surface
    """

    # Sanitize input

    # Get surface deformation info
    degu = ref_mesh.degree_u
    degv = ref_mesh.degree_v

    knotvec_u = ref_mesh.knotvector_u
    knotvec_v = ref_mesh.knotvector_v

    # Get control points of reference mesh
    ref_ctrlpts = np.array(ref_mesh.ctrlpts)

    num_ctrlpts = np.sqrt(len(ref_ctrlpts)).astype('int')

    # Deform control points
    def_ctrlpts = np.column_stack((ref_ctrlpts[:, 0] + cpts_disp[:, 0], ref_ctrlpts[:, 1] + cpts_disp[:, 1]))

    # Create deformed mesh
    def_mesh = bs.Surface()

    def_mesh.degree_u = degu
    def_mesh.degree_v = degv

    def_mesh.set_ctrlpts(def_ctrlpts.tolist(), num_ctrlpts, num_ctrlpts)

    def_mesh.knotvector_u = knotvec_u
    def_mesh.knotvector_v = knotvec_v

    return def_mesh


def mesh_znssd(ref_image, def_image, ref_mesh, cpts_disp, uv_vals=None, ref_coeff=None, def_coeff=None, interp_order='cubic'):
    """
    Compute the zero normalized sum of square differences over an entire mesh between two images
    See Pan et al. Meas Sci Tech 2009 for details.

    :param ref_image: reference image
    :type ref_image: ndarray
    :param def_image: deformed image
    :type def_image: ndarray
    :param ref_mesh: reference state B-spline mesh
    :type ref_mesh: Bspline surface
    :param cpts_disp: control point displacements [delta X, delta Y]
    :type cpts_disp: ndarray
    :param uv_vals: Optional. Array containing u, v parameterization of the pixels in the reference image
    :type uv_vals: ndarray
    :param ref_coeff: 2D array of b-spline coefficients for reference image
    :type ref_coeff: ndarray
    :param def_coeff: 2D array of b-spline coefficients for deformed image
    :type def_coeff: ndarray
    :param interp_order: order of b-spline interpolation. Options: {'cubic', 'quintic'}
    :type interp_order: str
    :return: ZNSSD between the two images in the mesh region
    :rtyp: float
    """

    # Sanitize input
    if ref_image.ndim != 2:
        raise ValueError('Reference image input must be 2d')

    if def_image.ndim != 2:
        raise ValueError('Deformed image input must be 2d')
    
    # Check if reference image interpolation coefficents have been set
    if not ref_coeff is None:
        if not ref_coeff.shape == ref_image.shape:
            raise ValueError('Coefficient array and image array must have same dimension (reference image)')
    else:
        ref_coeff = numerics.image_interp(ref_image, degree=interp_order)
    
    # Check if deformed image interpolation coefficeints have been set
    if not def_coeff is None:
        if not def_coeff.shape == def_image.shape:
            raise ValueError('Coefficient array and image array must have same dimension (deformed image)')
    else:
        def_coeff = numerics.image_interp(def_image, degree=interp_order)

    if ref_image.shape != def_image.shape:
        raise ValueError('Images must be the same size')

    # Get ref mesh control points
    ref_cpts = np.array(ref_mesh.ctrlpts)

    # Get def mesh 
    def_mesh = deform_mesh(ref_mesh, cpts_disp)

    # Get min and max column values from min/max reference ctrlpt node x values
    min_col_index = np.min(ref_cpts[:, 0]).astype('int')
    max_col_index = np.max(ref_cpts[:, 0]).astype('int')

    # Get maximum column number for sub image array from ref ctrlpt node x values
    colmax = (np.max(ref_cpts[:, 0]) - np.min(ref_cpts[:, 0])).astype('int')

    # Get min and max column values from min/max reference ctrlpt node x values
    min_row_index = np.min(ref_cpts[:, 1]).astype('int')
    max_row_index = np.max(ref_cpts[:, 1]).astype('int')

    # Get min and max row values from min/max reference ctrlpt node y values
    rowmax = (np.max(ref_cpts[:, 1]) - np.min(ref_cpts[:, 1])).astype('int')

    # Set reference image mesh over image
    f_mesh = ref_image[min_row_index:max_row_index, min_col_index:max_col_index]

    # Compute mean of this reference image mesh
    fmean = np.mean(f_mesh)

    # Compute standard deviation this reference image mesh
    fstddev = np.std(f_mesh)

    # For every pixel in f, compute the new location in g
    g_mesh = np.zeros(f_mesh.shape)

    # If uv_vals haven't been precomputed via projection, set them as a linear mapping of pixel coord to [0, 1]
    if uv_vals is None:
        uv_vals = np.zeros((2,) + f_mesh.shape)
        for i in range(0, rowmax):
            for j in range(0, colmax):
                uv_vals[0, i, j] = j / colmax
                uv_vals[1, i, j] = i / rowmax
    else:
        if not uv_vals.shape == ref_image.shape:
            raise ValueError('u, v parameterization array must be same shape as reference image')

    for i in range(0, rowmax):
        for j in range(0, colmax):
            u_val = uv_vals[0, i, j]
            v_val = uv_vals[1, i, j]

            # Compute the displacement by interpolating
            new_pt = def_mesh.surfpt(u_val, v_val)

            g_mesh[i, j] = numerics.eval_interp(new_pt[0], new_pt[1], def_image, coeffs=def_coeff, order=3)

    # Compute mean of this deformed image mesh
    gmean = np.mean(g_mesh)

    # Compute standard deviation of this deformed image mesh
    gstddev = np.std(g_mesh)

    # Loop over these matrices and compute ZNSSD (could be much faster)
    znssd = 0.0

    for i in range(0, rowmax):
        for j in range(0, colmax):
            znssd += np.square((f_mesh[i, j] - fmean) / fstddev - (g_mesh[i, j] - gmean) / gstddev)

    return znssd


def scipy_minfun(disp_vec, *args):

    '''
    Minimization function for passing to scipy minimize

    Assembles solution vector and arguments, then passes to mesh_znssd to compute cost

    :param disp_vec: trial displacement vector. Shape is (1, 2*number of mesh control points
    order is [delta x0, delta y0, delta x1, delta y1, etc.]
    :type disp_vec: ndarray
    :return: scalar value of mesh znssed at the trial displacement vector
    :rtype: float
    '''

    # Assemble displacement vector
    ctrlpt_disp = np.zeros((int(len(disp_vec) / 2), 2))
    for i in range(0, len(disp_vec), 2):
        k = i // 2  # Module to keep the index from over running lenght of control points
        ctrlpt_disp[k, :] = np.array([disp_vec[i], disp_vec[i + 1]])

    # Call znssd with defaults on all keyword params. This is slow, but okay for now
    znssd = analysis.mesh_znssd(*args, ctrlpt_disp)

    return znssd


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
    nodes_def = np.copy(nodes_ref)  # Copy
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
    :param def_image: deformed image array
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
