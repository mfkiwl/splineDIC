'''
.. script:: synth_def
    :platform: Unix, Windows
    :synopsis: Compute a NURBS DIC analysis between two images synthetically deformed.

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
from dic import visualize
import cv2
from matplotlib import pyplot as plt
import numpy as np
from geomdl import BSpline as bs
from geomdl import utilities as gutil
import scipy.optimize as sciopt
import scipy.signal as sig

# Debugging
import cProfile as profile

pr = profile.Profile()
pr.disable()

# Parse input
try:
    name = sys.argv[1]
    dx = float(sys.argv[2])
    dy = float(sys.argv[3])
    F11 = float(sys.argv[4])
    F12 = float(sys.argv[5])
    F21 = float(sys.argv[6])
    F22 = float(sys.argv[7])
except IndexError:
    print('Invalid command line arguments')
    sys.exit(1)

# Change to output directory
start = os.getcwd()
try:
    os.chdir(name)
except OSError:
    os.makedirs(name)
    os.chdir(name)

# Read image data
# Hard code absolute paths for now. Fix later'
dic_name = '/workspace/stpotter/git/bitbucket/dic/data/DIC_S_cropped_gray_pad_0.tiff'
psfdi_name = '/workspace/stpotter/git/bitbucket/dic/data/DOSA_cropped_gray_pad_0.tiff'
def_image = cv2.imread(dic_name, -1)  # Read in image 'as is'

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
ref_image = image_processing.im_warp(def_image, warp)

# Specify region of interest
# Format: [column index for start of X, column index for end of X, row index for start of Y, row index for end of Y]
subregion_indices = np.array([200, 300, 200, 300])

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

# Plot image with reference mesh nodes
x = coords[:, 0]
y = coords[:, 1]
fig, ax = plt.subplots(figsize=(10, 20))
ax.imshow(ref_image, cmap='gray')
ax.plot(x, y, 'o', color='red')
plt.savefig(name + 'mesh.png')

# Test synthetically deformed control points
synth_coords = np.zeros((len(coords), 2))
for i in range(len(synth_coords)):
    synth_coords[i, :] = np.matmul(F, coords[i, :]) + np.array([dx, dy])

# Compute synthetic control point displacements
synth_coords_disp = synth_coords - coords

# Compute znssd between synthetic and ref coordinates
synth_znssd = analysis.mesh_znssd(ref_image, def_image, ref_surf, synth_coords_disp)

# Print the synthetic info to stdout
print('Synthetic ZNSSD: {}'.format(synth_znssd))
print('Synthetic Coordinate Displacements')
print(synth_coords_disp)

# Visualize synthetic displacement results
# Set up new surface
disp_surf = bs.Surface()

disp_surf.degree_u = 3
disp_surf.degree_v = 3

disp_surf.set_ctrlpts(synth_coords_disp.tolist(), num_ctrlpts, num_ctrlpts)

disp_surf.knotvector_u = gutil.generate_knot_vector(disp_surf.degree_u, num_ctrlpts)
disp_surf.knotvector_v = gutil.generate_knot_vector(disp_surf.degree_v, num_ctrlpts)

disp_surf.delta = 0.01
fname = name + 'Synth'
visualize.viz_displacement(ref_image, disp_surf, rowmin, rowmax, colmin, colmax, fname)

fname = name + 'Synth'
# Visualize synthetic deformation results
visualize.viz_deformation(ref_image, ref_surf, rowmin, rowmax, colmin, colmax, synth_coords_disp, fname)

print('Deformation gradient at center of ROI from synthetic control points')
print(visualize.def_grad(ref_surf, 0.5, 0.5, synth_coords_disp))

# Wrap minimization arguments into a tuple
arg_tup = (ref_image, def_image, ref_surf)

# compute mesh znssd one time and exit if its low enough
int_disp_vec = analysis.rigid_guess(ref_image, def_image, rowmin, rowmax, colmin, colmax, len(coords))

# compute mesh znssd one time and exit if its low enough
#pr.enable()

residual = analysis.scipy_minfun(int_disp_vec, *arg_tup)

if residual > 1e-6:
    result = sciopt.minimize(analysis.scipy_minfun, int_disp_vec, args=arg_tup, method='L-BFGS-B', jac='2-point', bounds=None, options={'maxiter': 5, 'disp': True})

print('Actual Rigid X Displacement: {}'.format(dx))
print('Actual Rigid Y Displacement: {}'.format(dy))
print('Mesh Details: {} by {}'.format(num_ctrlpts, num_ctrlpts))
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

# Visualize minimization result displacement
# Set up new surface
disp_surf = bs.Surface()

disp_surf.degree_u = 3
disp_surf.degree_v = 3

disp_surf.set_ctrlpts(coords_disp.tolist(), num_ctrlpts, num_ctrlpts)

disp_surf.knotvector_u = gutil.generate_knot_vector(disp_surf.degree_u, num_ctrlpts)
disp_surf.knotvector_v = gutil.generate_knot_vector(disp_surf.degree_v, num_ctrlpts)

disp_surf.delta = 0.01

fname = name + 'Min'
visualize.viz_displacement(ref_image, disp_surf, rowmin, rowmax, colmin, colmax, fname)
# Visualize minimization result deformation
fname = name + 'Min'
visualize.viz_deformation(ref_image, ref_surf, rowmin, rowmax, colmin, colmax, coords_disp, fname)

print('Deformation gradient at center of ROI from minimization control points')
print(visualize.def_grad(ref_surf, 0.5, 0.5, coords_disp))

#pr.disable()
#pr.dump_stats('opt.pstat')
print('execution time (s)')

