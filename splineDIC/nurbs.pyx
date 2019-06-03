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

    if not N.flags['C_ CONTIGUOUS']:
        N = np.ascontiguousarray(N)

    if not knot_vector.flags['C_CONTIGUOUS']:
        knot_vector = np.ascontiguousarray(knot_vector)

    cdef double[::1] N_memview = N

    cdef double[::1] kv_memview = knot_vector

    error = basis_functionsC(&N_memview[0], knot_span, knot, degree, &kv_memview[0])

    if error != 0:
        print('Error calculating basis functions')
