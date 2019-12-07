cimport cython
import numpy as np

cdef extern from "../src/cnurbs.h":

    int find_spanC(unsigned int, unsigned int, double, double *)

def find_span(num_ctrlpts, degree, knot, knot_vector):

    if not knot_vector.flags['C_CONTIGUOUS']:
        knot_vector = np.ascontiguousarray(knot_vector)

    cdef double[::1] kv_memview = knot_vector

    span = find_spanC(num_ctrlpts, degree, knot, &kv_memview[0])

    return span

cdef extern from "../src/cnurbs.h":

    int basis_functionsC(double *, unsigned int, double, unsigned int, double *)

def basis_functions(N, knot_span, knot, degree, knot_vector):

    if not N.flags['C_CONTIGUOUS']:
        N = np.ascontiguousarray(N)

    if not knot_vector.flags['C_CONTIGUOUS']:
        knot_vector = np.ascontiguousarray(knot_vector)

    cdef double[::1] N_memview = N

    cdef double[::1] kv_memview = knot_vector

    error = basis_functionsC(&N_memview[0], knot_span, knot, degree, &kv_memview[0])

    if error != 0:
        print('Error calculating basis functions')

cdef extern from "../src/cnurbs.h":

    double surface_pointC(unsigned int, unsigned int, double *, unsigned int, unsigned int, double *, double *, double, double)

def surface_point(ncpts_u, deg_u, kv_u, ncpts_v, deg_v, kv_v, ctrlpts, u, v):

    if not kv_u.flags['C_CONTIGUOUS']:
        kv_u = np.ascontiguousarray(kv_u)

    if not kv_v.flags['C_CONTIGUOUS']:
        kv_v = np.ascontiguousarray(kv_v)

    if not ctrlpts.flags['C_CONTIGUOUS']:
        ctrlpts = np.ascontiguousarray(ctrlpts)

    cdef double[::1] kv_u_memview = kv_u

    cdef double[::1] kv_v_memview = kv_v

    cdef double[::1] ctrlpts_memview = ctrlpts

    s = surface_pointC(ncpts_u, deg_u, &kv_u_memview[0], ncpts_v, deg_v, &kv_v_memview[0], &ctrlpts_memview[0], u, v)

    return s

cdef extern from "../src/cnurbs.h":

    int surface_pointsC(double *, unsigned int, unsigned int, double *, unsigned int, unsigned int, double *, double *, double*, double*, unsigned int)

def surface_points(S, ncpts_u, deg_u, kv_u, ncpts_v, deg_v, kv_v, ctrlpts, u, v, npts):

    if not S.flags['C_CONTIGUOUS']:
        S = np.ascontiguousarray(S)

    if not kv_u.flags['C_CONTIGUOUS']:
        kv_u = np.ascontiguousarray(kv_u)

    if not kv_v.flags['C_CONTIGUOUS']:
        kv_v = np.ascontiguousarray(kv_v)

    if not ctrlpts.flags['C_CONTIGUOUS']:
        ctrlpts = np.ascontiguousarray(ctrlpts)

    if not u.flags['C_CONTIGUOUS']:
        u = np.ascontiguousarray(u)

    if not v.flags['C_CONTIGUOUS']:
        v = np.ascontiguousarray(v)

    cdef double[::1] S_memview = S

    cdef double[::1] kv_u_memview = kv_u

    cdef double[::1] kv_v_memview = kv_v

    cdef double[::1] ctrlpts_memview = ctrlpts

    cdef double[::1] u_memview = u

    cdef double[::1] v_memview = v

    error = surface_pointsC(&S_memview[0], ncpts_u, deg_u, &kv_u_memview[0], ncpts_v, deg_v, &kv_v_memview[0], &ctrlpts_memview[0], &u_memview[0], &v_memview[0], npts)

    if error != 0:
        print('Something went wrong computing the surface points!')
