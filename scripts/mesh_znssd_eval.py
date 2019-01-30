'''
.. script:: mesh_znssd_eval
    :platform: Unix, Windows
    :synopsis: Compute a NURBS DIC analysis between two images in single step

.. scriptauthor:: Sam Potter <spotter1642@gmail.com>
'''

# Path extensions (probably not necessary, but whatever)
# bootstrap $PATH
import sys
import os
sys.path.extend(['C:\\Users\\potterst1\\Desktop\Repositories\BitBucket\dic',
                 'C:/Users/potterst1/Desktop/Repositories/BitBucket/dic'])
sys.path.extend(['/workspace/stpotter/git/bitbucket/dic'])
import warnings
from dic import nurbs
from dic import fileIO
from dic import numerics
from dic import analysis
from dic import image_processing
import cv2
from matplotlib import pyplot as plt
import numpy as np
from geomdl import BSpline as bs
from geomdl import utilities as gutil
import scipy.optimize as sciopt

# Debugging
import cProfile as profile

pr = profile.Profile()
pr.disable()

# Read
# Hard code absolute paths for now. Fix later'
dic_name = 'C:\\Users\\potterst1\\Desktop\\Repositories\\BitBucket\\dic\\data\\DIC_S_cropped_gray_pad_0.tiff'
psfdi_name = 'C:\\Users\\potterst1\\Desktop\\Repositories\\BitBucket\\dic\\data\\DOA_cropped_gray_pad_0.tiff'
dic_name = '/workspace/stpotter/git/bitbucket/dic/data/DIC_S_cropped_gray_pad_0.tiff'
psfdi_name = '/workspace/stpotter/git/bitbucket/dic/data/DOSA_cropped_gray_pad_0.tiff'
ref_image = cv2.imread(dic_name, -1)  # Read in image 'as is'
ref_image = ref_image.astype('uint8')

# Translate image in x
transx = np.array([[1.0, 0.0, 5.0],
                   [0.0, 1.0, 0.0]])
def_image = image_processing.im_warp(ref_image, transx)

# Format: [column index for start of X, column index for end of X, row index for start of Y, row index for end of Y]
subregion_indices = np.array([225, 275, 225, 275])

# Control Points
rowmin = subregion_indices[-2:].min()
rowmax = subregion_indices[-2:].max()
colmin = subregion_indices[:2].min()
colmax = subregion_indices[:2].max()
x = np.linspace(colmin, colmax, 4)
y = np.linspace(rowmin, rowmax, 4)
coords = np.zeros((len(x) * len(y), 2))
k = 0
for i in range(0, len(x)):
    for j in range(0, len(y)):
        coords[k, :] = np.array([x[i], y[j]])
        k += 1

# Surface
ref_surf = bs.Surface()

ref_surf.degree_u = 3
ref_surf.degree_v = 3

num_ctrlpts = np.sqrt(len(coords)).astype('int')

ref_surf.set_ctrlpts(coords.tolist(), num_ctrlpts, num_ctrlpts)

ref_surf.knotvector_u = gutil.generate_knot_vector(ref_surf.degree_u, num_ctrlpts)
ref_surf.knotvector_v = gutil.generate_knot_vector(ref_surf.degree_v, num_ctrlpts)

ref_surf.delta = 0.01

arg_tup = (ref_image, def_image, ref_surf)

# Setup initial displacement vector
int_disp_vec = np.zeros((len(coords), 2))
for i in range(0, len(int_disp_vec)):
    int_disp_vec[i, 0] = 5.0
    int_disp_vec[i, 1] = 0.0

print('Ive called the mesh_znssd')
pr.enable()
result = analysis.mesh_znssd(ref_image, def_image, ref_surf, int_disp_vec)
pr.disable()
pr.dump_stats('mesh_znssd.pstat')
print(result)
