"""
.. module:: visualization
    :platform: Unix, Windows
    :synopsis: Module contains functions visualizing displacement and strain results

.. moduleauthor:: Sam Potter <spotter1642@gmail.com>
"""

from . import np
from . import helpers
from . import plt
from . import make_axes_locatable


def jac_inv(surf, u, v):

    """
    Compute inverse of jacobian given a NURBS surface and u, v parametric coordinates

    :param surf: NURBS surface
    :type surf: NURBS surface object
    :param u: u parametric location
    :type u: float
    :param v: v parametric location
    :type v: float
    :returns: computed inverse jacobian
    :rtype: ndarray
    """

    # Prefill matrix
    matrix = np.zeros((2, 2))

    # Compute surface derivatives
    derivs = surf.tangent((u, v), normalize=False)

    # Fill matrix
    # x1_u
    matrix[0, 0] = derivs[1][0]
    # x1_v
    matrix[0, 1] = derivs[1][1]
    # x2_u
    matrix[1, 0] = derivs[2][0]
    # x2_v
    matrix[1, 1] = derivs[2][1]

    # Compute inverse
    inv = np.linalg.inv(matrix)

    return inv


def R_u(deg_u, deg_v, kvec_u, kvec_v, span_u, span_v, u, v):

    """
    Compute the partial derivative of the surface basis function wrt parametric coordinates

    See Piegl & Tiller Eq. 3.17 for reference

    :param deg_u: degree of basis functions in u direction
    :type deg_u: int
    :param deg_v: degree of basis functions in v direction
    :type deg_v: int
    :param kvec_u: knot vector in u direction
    :type kvec_u: ndarray
    :param kvec_v: knot vector in v direction
    :type kvec_v: ndarray
    :param span_u: basis function number in u direction
    :type span_u: int
    :param span_v: basis function number in v direction
    :type span_v: int
    :param u: u parametric coordinate
    :type u: float
    :param v: v parametric coordinate
    :type v: float
    """

    # protoype function calls
    basis_der = helpers.basis_function_ders_one
    basis_one = helpers.basis_function_one

    # Get u derivative
    deriv_u = basis_der(deg_u, kvec_u, span_u, u, 1)[1] * basis_one(deg_v, kvec_v, span_v, v)

    # Get v derivative
    deriv_v = basis_one(deg_u, kvec_u, span_u, u) * basis_der(deg_v, kvec_v, span_v, v, 1)[1]

    return np.array([deriv_u, deriv_v])


def def_grad(surf, u, v, disp_vec):

    """
    Compute the components of the spatial derivatives of a NURBS mesh at a u, v location

    :param surf: NURBS surface on which to compute spatial derivatives
    :type surf: NURBS surface object
    :param u: u parametric coordinate
    :type u: float
    :param v: v parametric coordinate
    :type v: float
    :param disp_vec: array of control point displacements [delta x, delta y]
    :type disp_vec: ndarray
    :return: vector containing spatial derivatives of the NURBS surface evaluated at that coordinate location
    """

    # Compute J^-1
    Jinv = jac_inv(surf, u, v)

    # Pull out surface info
    deg_u = surf.degree_u
    deg_v = surf.degree_v
    kvec_u = surf.knotvector_u
    kvec_v = surf.knotvector_v
    ctrl_u = surf.ctrlpts_size_u
    ctrl_v = surf.ctrlpts_size_v

    F = np.zeros((2, 2))
    k = 0

    for spanu in range(0, ctrl_u):
        for spanv in range(0, ctrl_v):
            # Compute parametric derivatives
            dRdu = R_u(deg_u, deg_v, kvec_u, kvec_v, spanu, spanv, u, v)

            # Multiply with J^-1
            dRdx = np.matmul(dRdu, Jinv)

            # Fill F
            F[0, 0] += dRdx[0] * disp_vec[k, 0]  # d delta x1 dx1
            F[0, 1] += dRdx[1] * disp_vec[k, 0]  # d delta x1 dx2
            F[1, 0] += dRdx[0] * disp_vec[k, 1]  # d delta x2 dx1
            F[1, 1] += dRdx[1] * disp_vec[k, 1]  # d delta x2 dx2

            # Bump incrementer
            k += 1

            # Add identity tensor
            F += np.eye(2)

    return F


# Surface approach
def def_grad_surf(surf, u, v):
    """"
    Compute deformation gradient via a NURBS surface interpolation of control point displacement at u, v
    parametric coordinates

    :param surf: NURBS surface interpolating control point displacements
    :type surf: NURBS surface object
    :param u: u parametric location
    :type u: float
    :param v: v parametric location
    :type v: float
    :returns: computed inverse jacobian
    :rtype: ndarray
    """

    F = np.zeros((2, 2))

    tangents = surf.tangent((u, v), normalize=False)

    F[0, 0] = 1 + tangents[1][0]  # d delta x1 dx1
    F[0, 1] = tangents[1][1]  # d delta x1 dx2
    F[1, 0] = tangents[2][1]  # d delta x2 dx1
    F[1, 1] = 1 + tangents[2][1]  # d delta x2 dx2

    return F

def vis_displacement(ref_image, disp_surf, rowmin, rowmax, colmin, colmax, name, save=True):

    """
    Function to compute and visualize pixel level displacement from strain analysis.
    Displacement fields are plotted over the reference image

    :param ref_image: Reference image data.
    :type ref_image: ndarray
    :param disp_surf: Displacement surface where each control point encodes the x, y displacement value
    :type disp_surf: NURBS surface object
    :param rowmin: minimum row number of region of interest.
    :type rowmin: int
    :param rowmax: maximum row number of region of interest
    :type rowmax: int
    :param colmin: minimum column number of region of interest
    :type colmin: int
    :param colmax: maximum colum number of region of interest
    :type colmax: int
    :param name: name of analysis
    :type name: str
    :param save: Optional. Boolean for displaying or saving plot. Default True
    :type save: bool
    :return: None
    """

    # Fill x and y displacement arrays
    U = np.zeros(ref_image.shape) * np.nan
    V = np.zeros(ref_image.shape) * np.nan

    for i in range(rowmin, rowmax):
        for j in range(colmin, colmax):
            u_val = (j - colmin) / (colmax - colmin)
            v_val = (i - rowmin) / (rowmax - rowmin)
            disp = disp_surf.surfpt(u_val, v_val)
            U[i, j] = disp[0]
            V[i, j] = disp[1]

    # Display
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 10))
    im0 = ax0.imshow(ref_image, cmap='gray')
    Uim = ax0.imshow(U, cmap='jet', alpha=0.7)
    divider = make_axes_locatable(ax0)
    cax0 = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(Uim, cax=cax0)
    Umin = 0.9 * np.nanmin(U)
    Umax = 1.1 * np.nanmax(U)
    Uim.set_clim(Umin, Umax)
    ax0.set_title('X Displacement (Pixels)')

    im1 = ax1.imshow(ref_image, cmap='gray')
    Vim = ax1.imshow(V, cmap='jet', alpha=0.7)
    divider = make_axes_locatable(ax1)
    cax1 = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(Vim, cax=cax1)
    Vmin = 0.9 * np.nanmin(V)
    Vmax = 0.9 * np.nanmax(V)
    Vim.set_clim(Vmin, Vmax)
    ax1.set_title('Y Displacement (Pixels)')

    if save:
        plt.savefig(name + 'Displacements.png')
    else:
        plt.show()
