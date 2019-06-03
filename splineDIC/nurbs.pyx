cimport cython
import numpy as np

cdef extern from "../src/nurbs.c":

    pass

cdef extern from "../src/nurbs.h":

    int find_span(unsigned int, unsigned int, double, double *)

def find_span(num_ctrlpts, degree, knot, knot_vector):

    if not knot_vector.flags['C_CONTIGUOUS']:
        knot_vector = np.ascontiguousarray(knot_vector)

    cdef double[::1] kv_memview = knot_vector

    span = find_span(num_ctrlpts, degree, knot, &kv_memview[0])

    return span