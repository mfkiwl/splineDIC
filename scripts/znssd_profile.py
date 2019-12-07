"""
.. script:: znssd_profile
    :platform: Unix, Windows
    :synopsis: Compute synthetic ZNSSD for profiling and timing purposes

.. scriptauthor:: Sam Potter <spotter1642@gmail.com>
"""

import sys
import os

sys.path.extend(['C:\\Users\\potterst1\\Desktop\Repositories\BitBucket\dic',
                 'C:/Users/potterst1/Desktop/Repositories/BitBucket/dic'])
sys.path.extend(['/workspace/stpotter/git/bitbucket/dic'])
from dic import numerics
from dic import analysis
import cv2
import numpy as np
from geomdl import BSpline as bs
from geomdl import utilities as gutil

# Debugging
import cProfile as profile
import pdb

pr = profile.Profile()
pr.disable()

# Parse input
data = 'dic'
dx = 0
dy = 0
F11 = 1.01
F12 = 0
F21 = 0
F22 = 1

# Read image data
# Hard code absolute paths for now. Fix later'
dic_name = '/workspace/stpotter/git/bitbucket/dic/data/DIC_S_cropped_gray_pad_0.tiff'
psfdi_name = '/workspace/stpotter/git/bitbucket/dic/data/DOA_cropped_gray_pad_0.tiff'

if data == 'dic':
    def_image = cv2.imread(dic_name, -1)  # Read in image 'as is'
elif data == 'psfdi':
    def_image = cv2.imread(psfdi_name, -1)  # Read in image 'as is'
else:
    print('Unclear image data type')
    sys.exit(1)

# Translate image
F = np.array([[F11, F12],
              [F21, F22]])

# get def image interp coefficients
def_coeff = numerics.image_interp_bicubic(def_image)

# Create sub images (def)
def_sub_image = def_image[:450, :450]

# Create ref sub image
ref_sub_image = np.zeros(def_sub_image.shape)
i = 0
j = 0
for row in range(0, 450):
    for col in range(0, 450):
        # New pt (x, y)
        pt = F @ np.array([col, row]) + np.array([dx, dy])
        val = numerics.eval_interp_bicubic(def_coeff, pt[0], pt[1], def_image.shape)
        ref_sub_image[i, j] = val
        j += 1
    j = 0
    i += 1

# Specify region of interest
# Format: [column index for start of X, column index for end of X, row index for start of Y, row index for end of Y]
# TODO: Is this format the best or should it be row column and then map appropriately? Depends on UI
subregion_indices = np.array([50, 425, 50, 425])

# Compute some items to use for output. Will refactor later

# Setup mesh and uv values
ref_surf, uv_vals, coords, indices = analysis.setup_surf(subregion_indices)
num_ctrlpts = np.sqrt(len(coords)).astype('int')

# Get interpolation coefficients
ref_sub_coeff = numerics.image_interp_bicubic(ref_sub_image)
def_sub_coeff = numerics.image_interp_bicubic(def_sub_image)

# TODO: Add type checking

# Compute reference mesh quantities of interest (array, mean, standard deviation)
f_mesh, f_mean, f_stddev = analysis.ref_mesh_qoi(ref_surf, uv_vals, ref_sub_coeff, ref_sub_image.shape)

# Test synthetically deformed control points
synth_coords = np.zeros((len(coords), 2))
for i in range(len(synth_coords)):
    synth_coords[i, :] = np.matmul(F, coords[i, :]) + np.array([dx, dy])

# Compute synthetic control point displacements
synth_coords_disp = synth_coords - coords

# Compute znssd between synthetic and ref coordinates
pr.enable()
synth_znssd = analysis.mesh_znssd(f_mesh, f_mean, f_stddev, def_sub_image.shape, ref_surf, uv_vals, def_sub_coeff,
                                  synth_coords_disp)

pr.disable()
pr.dump_stats('znssd.pstat')
