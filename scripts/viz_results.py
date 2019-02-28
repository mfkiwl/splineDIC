'''
.. script:: viz_results
    :platform: Unix, Windows
    :synopsis: Visualize displacement and deformation results from DIC outputs

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
import pdb

pr = profile.Profile()
pr.disable()

# Parse input
try:
    impath = sys.argv[1]
    datapath = sys.argv[2]
except IndexError:
    print('Invalid command line arguments')
    sys.exit(1)

# Change to data directory
start = os.getcwd()
try:
    os.chdir(datapath)
except OSError:
    print('Path does not exist')
    sys.exit(1)

# Load image as is
image = cv2.imread(impath, -1)

# Cut off last part of data path name to get data set name
dsetname = os.path.split(datapath)[1]

# Open up synthetic mesh coordinates
fname = dsetname + 'synthdef.txt'
f = open(fname, 'r')
lines = f.readlines()
lines = lines[2:]
synth_data = np.zeros((len(lines), 4))
for i in range(0, len(synth_data)):
    data = list(map(float, lines[i].split()))
    synth_data[i, :] = np.array(data)

f.close()

# Open up minimization mesh coordinates
fname = dsetname + 'mindef.txt'
f = open(fname, 'r')
lines = f.readlines()
lines = lines[2:]
min_data = np.zeros((len(lines), 4))
for i in range(0, len(synth_data)):
    data = list(map(float, lines[i].split()))
    min_data[i, :] = np.array(data)

f.close()

# Grad coordinates from synth mesh (should be same as min mesh)
coords = synth_data[:, :2]

# Setup ref surface
# Hard code mesh details for now
# Surface
ref_surf = bs.Surface()

ref_surf.degree_u = 3
ref_surf.degree_v = 3

num_ctrlpts = np.sqrt(len(coords)).astype('int')

ref_surf.set_ctrlpts(coords.tolist(), num_ctrlpts, num_ctrlpts)

ref_surf.knotvector_u = gutil.generate_knot_vector(ref_surf.degree_u, num_ctrlpts)
ref_surf.knotvector_v = gutil.generate_knot_vector(ref_surf.degree_v, num_ctrlpts)

ref_surf.delta = 0.01

# Plot mesh details
x = coords[:, 0]
y = coords[:, 1]
fig, ax = plt.subplots(figsize=(10, 20))
ax.imshow(image, cmap='gray')
ax.plot(x, y, 'o', color='red')
plt.savefig(dsetname + 'mesh.png')

rowmin_index = int(synth_data[:, 1].min())
rowmax_index = int(synth_data[:, 1].max())
colmin_index = int(synth_data[:, 0].min())
colmax_index = int(synth_data[:, 0].max())

# Compute ROI and ROI uv values
# Get min and max column values from min/max reference ctrlpt node x values
min_col_index = np.min(coords[:, 0]).astype('int')
max_col_index = np.max(coords[:, 0]).astype('int')

# Get maximum column number for sub image array from ref ctrlpt node x values
colmax = (np.max(coords[:, 0]) - np.min(coords[:, 0])).astype('int')

# Get min and max column values from min/max reference ctrlpt node x values
min_row_index = np.min(coords[:, 1]).astype('int')
max_row_index = np.max(coords[:, 1]).astype('int')

# Get min and max row values from min/max reference ctrlpt node y values
rowmax = (np.max(coords[:, 1]) - np.min(coords[:, 1])).astype('int')

# Set reference image mesh over image
roi = image[min_row_index:max_row_index, min_col_index:max_col_index]

uv_vals = np.zeros((2, )+ roi.shape)
for i in range(0, rowmax):
    for j in range(0, colmax):
        uv_vals[0, i, j] = j / colmax
        uv_vals[1, i, j] = i / rowmax

# Visualize synthetic displacement results
synth_coords_disp = synth_data[:, 2:]
# Set up new surface
disp_surf = bs.Surface()

disp_surf.degree_u = 3
disp_surf.degree_v = 3

disp_surf.set_ctrlpts(synth_coords_disp.tolist(), num_ctrlpts, num_ctrlpts)

disp_surf.knotvector_u = gutil.generate_knot_vector(disp_surf.degree_u, num_ctrlpts)
disp_surf.knotvector_v = gutil.generate_knot_vector(disp_surf.degree_v, num_ctrlpts)

disp_surf.delta = 0.01
fname = dsetname + 'Synth'
visualize.viz_displacement(image, disp_surf, rowmin_index, rowmax_index, colmin_index, colmax_index, fname)

fname = dsetname + 'Synth'
# Visualize synthetic deformation results
visualize.viz_deformation(image, ref_surf, rowmin_index, rowmax_index, colmin_index, colmax_index,
                          synth_coords_disp, fname)


