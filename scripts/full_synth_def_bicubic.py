'''
.. script:: full_synth_def_bicubic
    :platform: Unix, Windows
    :synopsis: Compute a NURBS DIC analysis between two images synthetically generated images using bicubic
    interpolation methods

.. scriptauthor:: Sam Potter <spotter1642@gmail.com>
'''

# Path extensions (probably not necessary, but whatever)
# bootstrap $PATH
import sys
import os

sys.path.extend(['C:\\Users\\potterst1\\Desktop\Repositories\BitBucket\dic',
                 'C:/Users/potterst1/Desktop/Repositories/BitBucket/dic'])
sys.path.extend(['/workspace/stpotter/git/bitbucket/dic'])
from dic import numerics
from dic import analysis
from dic import visualize
import cv2
import numpy as np
from geomdl import BSpline as bs
from geomdl import utilities as gutil
import scipy.optimize as sciopt

# Debugging
import cProfile as profile
import pdb

pr = profile.Profile()
pr.disable()

# Parse input
try:
    system = sys.argv[1]
    data = sys.argv[2]
    name = sys.argv[3]
    maxiterations = int(sys.argv[4])
    dx = float(sys.argv[5])
    dy = float(sys.argv[6])
    F11 = float(sys.argv[7])
    F12 = float(sys.argv[8])
    F21 = float(sys.argv[9])
    F22 = float(sys.argv[10])
except IndexError:
    print('Invalid command line arguments')
    sys.exit(1)

# Change to output directory
start = os.getcwd()
dirname = str(maxiterations) + 'Iters'
try:
    os.chdir(dirname)
except OSError:
    os.makedirs(dirname)
    os.chdir(dirname)
try:
    os.chdir(name)
except OSError:
    os.makedirs(name)
    os.chdir(name)

# Read image data
# Hard code absolute paths for now. Fix later'
if system == 'windows':
    dic_name = 'C:\\Users\\potterst1\\Desktop\\Repositories\\BitBucket\\dic\\data\\DIC_S_cropped_gray_pad_0.tiff'
    psfdi_name = 'C:\\Users\\potterst1\\Desktop\\Repositories\\BitBucket\\dic\\data\\DOA_cropped_gray_pad_0.tiff'
elif system == 'unix':
    dic_name = '/workspace/stpotter/git/bitbucket/dic/data/DIC_S_cropped_gray_pad_0.tiff'
    psfdi_name = '/workspace/stpotter/git/bitbucket/dic/data/DOA_cropped_gray_pad_0.tiff'
else:
    print('Unclear system specification')
    sys.exit(1)

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
Finv = np.linalg.inv(F)
F11i = Finv[0, 0]
F12i = Finv[0, 1]
F21i = Finv[1, 0]
F22i = Finv[1, 1]
warp = np.array([[F11i, F12i, -dx],
                 [F21i, F22i, -dy]])

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
subregion_indices = np.array([100, 400, 100, 400])

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
synth_znssd = analysis.mesh_znssd(f_mesh, f_mean, f_stddev, def_sub_image.shape, ref_surf, uv_vals, def_sub_coeff,
                                  synth_coords_disp)

# Print the synthetic info to stdout
data_out = 'Using image data from ' + data
print(data_out)
print('Synthetic ZNSSD: {}'.format(synth_znssd))
print('Synthetic Coordinate Displacements')
print(synth_coords_disp)
print('Deformation gradient at center of ROI from synthetic control points')
print(visualize.def_grad(ref_surf, 0.5, 0.5, synth_coords_disp))

# Wrap minimization arguments into a tuple
arg_tup = (f_mesh, f_mean, f_stddev, def_sub_image.shape, ref_surf, uv_vals, def_sub_coeff)

# compute mesh znssd one time and exit if its low enough
int_disp_vec = analysis.rigid_guess(ref_sub_image, def_sub_image, indices[0], indices[1], indices[2], indices[3],
                                    len(coords))

# compute mesh znssd one time and exit if its low enough
#pr.enable()

residual = analysis.scipy_minfun(int_disp_vec, *arg_tup)
minoptions = {'maxiter': 30, 'disp': False}

if residual > 1e-6:
    result = sciopt.minimize(analysis.scipy_minfun, int_disp_vec, args=arg_tup, method='L-BFGS-B', jac='2-point',
                             bounds=None, options=minoptions)

print('Actual Rigid X Displacement: {}'.format(dx))
print('Actual Rigid Y Displacement: {}'.format(dy))
print('Mesh Details: {} by {}'.format(num_ctrlpts, num_ctrlpts))
print('ROI Size: {} by {}'.format(indices[1] - indices[0], indices[3] - indices[2]))
print('Initial Guess -  X Displacement: {}'.format(int_disp_vec[0]))
print('Initial Guess - Y Displacement: {}'.format(int_disp_vec[1]))

if residual > 1e-6:
    print('residual')
    print(result.fun)
    coords_disp = np.column_stack((result.x[::2], result.x[1::2]))
    print('final control point displacements')
    print(coords_disp)
else:
    print('residual')
    print(residual)
    coords_disp = np.column_stack((int_disp_vec[::2], int_disp_vec[1::2]))
    print('final control point displacement')
    print(coords_disp)

print('Deformation gradient at center of ROI from minimization control points')
print(visualize.def_grad(ref_surf, 0.5, 0.5, coords_disp))

# Write outputs
fname = name + 'synthdef.txt'
f = open(fname, 'w')
f.write('Mesh Coordinates\n')
f.write('X Y dX dY\n')
for i in range(0, len(coords)):
    f.write('{0} {1} {2} {3} \n'.format(coords[i, 0], coords[i, 1], synth_coords_disp[i, 0], synth_coords_disp[i, 1]))

f.close()

fname = name + 'mindef.txt'
f = open(fname, 'w')
f.write('Mesh Coordinates\n')
f.write('X Y dX dY\n')
for i in range(0, len(coords)):
    f.write('{0} {1} {2} {3} \n'.format(coords[i, 0], coords[i, 1], coords_disp[i, 0], coords_disp[i, 1]))

f.close()
