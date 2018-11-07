""" Finite element DIC

.. moduleauthor:: Sam POtter <spotter1642@gmail.com>

"""

__author__ = "Sam Potter"
__version__ = "1.0"
__liscens__ = "MIT"

import numpy as np
from scipy import io as sio
import os
import cv2
from matplotlib import pyplot as plt
from matplotlib import cm
import errno
import regex
from scipy import signal