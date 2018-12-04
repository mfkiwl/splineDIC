'''

    Tests for nurbs module of the DIC Module
    Released under MIT License. See LICENSE file for details
    Copyright (C) 2018 Sam Potter

    Requires pytest
'''

from .context import dic
from dic import nurbs

import pytest
import numpy as np


def test_basis_functions():

    degree = 2
    knot_vector = [0, 0, 0, 1, 2, 3, 4, 4, 5, 5, 5]
    # n = m - p - 1 -> n + 1 = m + 1 - p - 1

    # The NURBS Book Ex. 2.3
    basis_vals = nurbs.basis_functions(4, 5.0/2.0, degree, knot_vector)

    expected = np.array([0.125, 0.75, 0.125])

    condition = np.allclose(basis_vals, expected)

    assert condition


def test_basis_functions2():

    degree = 2
    knot_vector = [0, 0, 0, 1, 2, 3, 4, 4, 5, 5, 5]
    # n = m - p - 1 -> n + 1 = m + 1 - p - 1

    # The NURBS Book Ex. 2.3
    basis_vals = nurbs.basis_functions(4, 5.0/2.0, degree, knot_vector)

    basis_sum = np.sum(basis_vals)

    assert np.isclose(basis_sum, 1.0)

# TODO: Add tests for:
# TODO: Add find_span_binary
# TODO: curve_point
# TODO: curve_ders
# TODO: check_knot_vec
# TODO: generate_knot_vec

# TODO: Add basis_functions_one
# TODO: Add basis_functions_ders_one
