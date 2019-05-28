cimport cython

cdef extern from "../src/nurbs.h":

    void find_span()

def find_span(num_ctrlpts, degree, knot, knot_vector):

    find_span(num_ctrlpts, degree, knot, knot_vector)