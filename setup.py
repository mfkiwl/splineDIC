#!/usr/bin/env python

"""
    Simple setup.py script for splineDIC package
"""

from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
import numpy as np

# Meta-data
NAME = 'splinedic'
DESCRIPTION = 'Digital image correlation using spline basis functions'
URL = 'https://github.com/stpotter16/splineDIC'
EMAIL = 'spotter1642@gmail.com'
AUTHOR = 'Sam Potter'
REQUIRES_PYTHON = '>=3.6.0'
VERSION = 1.0
REQUIRED = ['numpy', 'scipy', 'opencv-python', 'matplotlib', 'regex', 'numba', 'cython']

# Cython
ext = Extension('nurbs', ['src/cnurbs.c', 'splineDIC/nurbs.pyx'], include_dirs=[np.get_include()])

# Call setup
setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    py_modules=['splineDIC'],
    ext_modules=cythonize(ext),
    install_requires=REQUIRED,
    license='MIT',
    packages=find_packages()
)
