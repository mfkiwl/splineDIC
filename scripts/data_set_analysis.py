'''
.. script:: data_set_analysis
    :platform: Unix, Windows
    :synopsis: Compute a NURBS DIC analysis on a set of experimental data. Uses bicubic interpolation

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
    folder = sys.argv[1]
except OSError:
    print('Invalid command line arguments')
    sys.exit(1)

# Change to data directory
start = os.getcwd()
try:
    os.chdir(folder)
except OSError:
    print('Invalid directory')
    sys.exit(1)

# Expect TIFF files for now
# TODO: Make less brittle

files = os.listdir()
images = [name for name in files if os.path.splitext(name)[1] == '.tiff']

# Sort images. Expects them to be numerically labeled.
# TODO: Make less brittle
images.sort()

# Specify subregion indices
# TODO: Make input
# Format: [column index for start of X, column index for end of X, row index for start of Y, row index for end of Y]
subregion_indices = np.array([75, 475, 100, 500])

# Main analysis loop
for step in range(0, len(images) - 1):
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
    ref_image = cv2.imread(images[step], -1)
    def_image = cv2.imread(images[step + 1], -1)

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

    # Write outputs of step to file

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
f.write('Using data from {}\n'.format(folder))
f.write('Reference Image: {}\n'.format(images[0]))
f.write('Final Image: {}\n'.format(images[-1]))
f.write('Mesh Details: {} by {}\n'.format(num_ctrlpts, num_ctrlpts))
f.write('ROI Size: {} by {}\n'.format(indices[1] - indices[0], indices[3] - indices[2]))

f.close()
