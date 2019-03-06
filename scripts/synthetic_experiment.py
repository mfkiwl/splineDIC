'''
.. script:: synthetic_experiment
    :platform: Unix, Windows
    :synopsis: Compute a NURBS DIC analysis on a set of synthetically generated experimental images

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
    numsteps = int(sys.argv[4])
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
dirname = str(numsteps) + 'Steps'
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

# Generate set of deformation gradients
F = np.array([[F11, F12],
              [F21, F22]])

rigid = np.array([dx, dy])

F_set = np.zeros((numsteps,) + (4, 4))
rigid_set = np.zeros((numsteps,) + (1, 2))

F_set[0, :, :] = F
rigid_set[0, :, :] = rigid

temp_F = F
temp_rigid = rigid
for i in range(1, numsteps):
    temp_F = temp_F @ F
    temp_rigid = temp_rigid + rigid

    F_set[i, :, :] = temp_F
    rigid_set[i, :, :] = temp_rigid

# get def image interp coefficients
def_coeff = numerics.image_interp_bicubic(def_image)

# Create sub images (def)
def_sub_image = def_image[:450, :450]

# Create ref sub image and coords
ref_sub_images = np.zeros((numsteps,) + def_sub_image.shape)
for step in range(0, numsteps):
    F = F_set[step, :, :]
    dxdy = rigid_set[step, :, :]
    i = 0
    j = 0
    for row in range(0, 450):
        for col in range(0, 450):
            # New pt (x, y)
            pt = F @ np.array([col, row]) + dxdy
            val = numerics.eval_interp_bicubic(def_coeff, pt[0], pt[1], def_image.shape)
            ref_sub_images[i, j] = val
            j += 1
        j = 0
        i += 1

# Specify region of interest
# Format: [column index for start of X, column index for end of X, row index for start of Y, row index for end of Y]
# TODO: Is this format the best or should it be row column and then map appropriately? Depends on UI
subregion_indices = np.array([100, 400, 100, 400])

# Main analysis loop
for step in range(0, numsteps - 1):
    # Setup analysis if needed, else pass results from previous step
    if step == 0:
        mesh_surf, uv_vals, coords, indices = analysis.setup_surf(subregion_indices)
        num_ctrlpts = np.sqrt(len(coords)).astype('int')
    else:
        # Compute new coordinate locations
        coords = np.array(mesh_surf.ctrlpts) + coords_disp

        # Set mesh with new control points
        mesh_surf.set_ctrlpts(coords.tolist(), num_ctrlpts, num_ctrlpts)

    # Open images
    if step == numsteps -2:
        ref_image = ref_sub_images[step, :, :]
        def_image = def_sub_image
    else:
        ref_image = ref_sub_images[step, :, :]
        def_image = ref_sub_images[step + 1, :, :]

    # Interpolate images
    ref_coeff = numerics.image_interp_bicubic(ref_image)
    def_coeff = numerics.image_interp_bicubic(def_image)

    # Compute reference mesh quantities of interest (array, mean, standard deviation)
    f_mesh, f_mean, f_stddev = analysis.ref_mesh_qoi(mesh_surf, uv_vals, ref_coeff, ref_image.shape)

    # Wrap minimization arguments into a tuple
    arg_tup = (f_mesh, f_mean, f_stddev, def_image.shape, mesh_surf, uv_vals, def_coeff)

    # Compute rigid guess
    int_disp_vec = analysis.rigid_guess(ref_image, def_image, indices[0], indices[1], indices[2], indices[3],
                                        len(coords))

    # Setup minimization options
    minoptions = {'maxiter': 20, 'disp': False}

    # Minimize
    minresults = sciopt.minimize(analysis.scipy_minfun, int_disp_vec, args=arg_tup, method='L-BFGS-B', jac='2-point',
                             bounds=None, options=minoptions)

    coords_disp = np.column_stack((minresults.x[::2], minresults.x[1::2]))

    # Synthetic displacement results
    synth_coords = np.zeros((len(coords), 2))
    for i in range(len(synth_coords)):
        synth_coords[i, :] = F @ coords[i, :] + rigid

    # Compute synthetic control point displacements
    synth_coords_disp = synth_coords - coords

    # Compute znssd between synthetic and ref coordinates
    synth_znssd = analysis.mesh_znssd(f_mesh, f_mean, f_stddev, def_sub_image.shape, mesh_surf, uv_vals, def_sub_coeff,
                                      synth_coords_disp)

    # Write outputs of step to file
    fname = name + 'synthdef.txt'
    f = open(fname, 'w')
    f.write('Mesh Coordinates\n')
    f.write('X Y dX dY\n')
    for i in range(0, len(coords)):
        f.write('{0} {1} {2} {3} \n'.format(coords[i, 0], coords[i, 1], synth_coords_disp[i, 0],
                                            synth_coords_disp[i, 1]))

    f.close()

    fname = str(step) + str(step + 1) + 'results.txt'
    f = open(fname, 'w')
    f.write('Final Minimization ZNSSD: {}\n'.format(minresults.fun))
    f.write('Mesh Coordinates\n')
    f.write('X Y dX dY\n')
    for i in range(0, len(coords)):
        f.write('{0} {1} {2} {3} \n'.format(coords[i, 0], coords[i, 1], coords_disp[i, 0], coords_disp[i, 1]))

    f.close()

# Write analysis details
fname = 'analysis_summary.txt'
f = open(fname, 'w')
f.write('Mesh Details: {} by {}\n'.format(num_ctrlpts, num_ctrlpts))
f.write('ROI Size: {} by {}\n'.format(indices[1] - indices[0], indices[3] - indices[2]))

f.close()
