'''
.. script:: error_check
    :platform: Unix, Windows
    :synopsis: Viz displacement and deformation results and compute differences between synthetic and minimization
    results and compute statistics on diffs

.. scriptauthor:: Sam Potter <spotter1642@gmail.com>
'''

# Path extensions (probably not necessary, but whatever)
# bootstrap $PATH
import sys
import os

sys.path.extend(['C:\\Users\\potterst1\\Desktop\Repositories\BitBucket\dic',
                 'C:/Users/potterst1/Desktop/Repositories/BitBucket/dic'])
sys.path.extend(['/workspace/stpotter/git/bitbucket/dic'])
from dic import visualize
import cv2
from matplotlib import pyplot as plt
import numpy as np
from geomdl import BSpline as bs
from geomdl import utilities as gutil
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import stats

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
synth_disp_surf = bs.Surface()

synth_disp_surf.degree_u = 3
synth_disp_surf.degree_v = 3

synth_disp_surf.set_ctrlpts(synth_coords_disp.tolist(), num_ctrlpts, num_ctrlpts)

synth_disp_surf.knotvector_u = gutil.generate_knot_vector(synth_disp_surf.degree_u, num_ctrlpts)
synth_disp_surf.knotvector_v = gutil.generate_knot_vector(synth_disp_surf.degree_v, num_ctrlpts)

synth_disp_surf.delta = 0.01
fname = dsetname + 'Synth'
visualize.viz_displacement(image, synth_disp_surf, rowmin_index, rowmax_index, colmin_index, colmax_index, fname)

fname = dsetname + 'Synth'
# Visualize synthetic deformation results
visualize.viz_deformation(image, ref_surf, rowmin_index, rowmax_index, colmin_index, colmax_index,
                          synth_coords_disp, fname)

# Visualize minimization displacement results
min_coords_disp = min_data[:, 2:]
# Set up new surface
min_disp_surf = bs.Surface()

min_disp_surf.degree_u = 3
min_disp_surf.degree_v = 3

min_disp_surf.set_ctrlpts(min_coords_disp.tolist(), num_ctrlpts, num_ctrlpts)

min_disp_surf.knotvector_u = gutil.generate_knot_vector(min_disp_surf.degree_u, num_ctrlpts)
min_disp_surf.knotvector_v = gutil.generate_knot_vector(min_disp_surf.degree_v, num_ctrlpts)

min_disp_surf.delta = 0.01
fname = dsetname + 'Min'
visualize.viz_displacement(image, min_disp_surf, rowmin_index, rowmax_index, colmin_index, colmax_index, fname)

fname = dsetname + 'Min'
# Visualize synthetic deformation results
visualize.viz_deformation(image, ref_surf, rowmin_index, rowmax_index, colmin_index, colmax_index,
                          min_coords_disp, fname)

# Compute differences in displacement
# Visualize differences in displacement
# TODO: Make this a function
# Fill x and y displacement arrays
U_diff = np.zeros(image.shape) * np.nan
V_diff = np.zeros(image.shape) * np.nan

for i in range(rowmin_index, rowmax_index):
    for j in range(colmin_index, colmax_index):
        u_val = (j - colmin_index) / (colmax_index - colmin_index)
        v_val = (i - rowmin_index) / (rowmax_index - rowmin_index)
        disp_diff = np.array(synth_disp_surf.surfpt(u_val, v_val)) - np.array(min_disp_surf.surfpt(u_val, v_val))
        U_diff[i, j] = disp_diff[0]
        V_diff[i, j] = disp_diff[1]

# Display
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 10))
im0 = ax0.imshow(image, cmap='gray')
Uim = ax0.imshow(U_diff, cmap='jet', alpha=0.7)
divider = make_axes_locatable(ax0)
cax0 = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(Uim, cax=cax0)
Umin = 0.9 * np.nanmin(U_diff)
Umax = 1.1 * np.nanmax(U_diff)
Uim.set_clim(Umin, Umax)
ax0.set_title('X Displacement (Pixels)')

im1 = ax1.imshow(image, cmap='gray')
Vim = ax1.imshow(V_diff, cmap='jet', alpha=0.7)
divider = make_axes_locatable(ax1)
cax1 = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(Vim, cax=cax1)
Vmin = 0.9 * np.nanmin(V_diff)
Vmax = 0.9 * np.nanmax(V_diff)
Vim.set_clim(Vmin, Vmax)
ax1.set_title('Y Displacement (Pixels)')

plt.savefig('Displacements Differences.png')

# Statistics on difference in displacement

U_diff_mean = np.nanmean(U_diff)
V_diff_mean = np.nanmean(V_diff)

U_SEM = stats.sem(U_diff, axis=None)
V_SEM = stats.sem(V_diff, axis=None)

# Compute differences in deformation
# Viaualize differences in deformation
# TODO: Make this a function
# Fill x and y displacement arrays
F11_diff = np.zeros(image.shape) * np.nan
F12_diff = np.zeros(image.shape) * np.nan
F21_diff = np.zeros(image.shape) * np.nan
F22_diff = np.zeros(image.shape) * np.nan

for i in range(rowmin_index, rowmax_index):
    for j in range(colmin_index, colmax_index):
        u_val = (j - colmin_index) / (colmax_index - colmin_index)
        v_val = (i - rowmin_index) / (rowmax_index - rowmin_index)
        F_diff = visualize.deg_grad(ref_surf, u_val, v_val, synth_coords_disp) - visualize.def_grad(ref_surf, u_val, v_val, min_coords_disp)
        F11_diff[i, j] = F_diff[0, 0]
        F12_diff[i, j] = F_diff[0, 1]
        F21_diff[i, j] = F_diff[1, 0]
        F22_diff[i, j] = F_diff[1, 1]

# Display
fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2, figsize=(15, 10))

im0 = ax0.imshow(image, cmap='gray')
F11im = ax0.imshow(F11_diff, cmap='jet', alpha=0.7)
divider = make_axes_locatable(ax0)
cax0 = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(F11im, cax=cax0)
F11min = 0.9 * np.nanmin(F11_diff)
F11max = 1.1 * np.nanmax(F11_diff)
F11im.set_clim(F11min, F11max)
ax0.set_title('F11')

im1 = ax1.imshow(image, cmap='gray')
F12im = ax1.imshow(F12_diff, cmap='jet', alpha=0.7)
divider = make_axes_locatable(ax1)
cax1 = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(F12im, cax=cax1)
F12min = 0.9 * np.nanmin(F12_diff)
F12max = 1.1 * np.nanmax(F12_diff)
F12im.set_clim(F12min, F12max)
ax1.set_title('F12')

im2 = ax2.imshow(image, cmap='gray')
F21im = ax2.imshow(F21_diff, cmap='jet', alpha=0.7)
divider = make_axes_locatable(ax2)
cax2 = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(F21im, cax=cax2)
F21min = 0.9 * np.nanmin(F21_diff)
F21max = 1.1 * np.nanmax(F21_diff)
F21im.set_clim(F21min, F21max)
ax2.set_title('F21')

im3 = ax3.imshow(image, cmap='gray')
F22im = ax3.imshow(F22_diff, cmap='jet', alpha=0.7)
divider = make_axes_locatable(ax3)
cax3 = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(F22im, cax=cax3)
F22min = 0.9 * np.nanmin(F22_diff)
F22max = 1.1 * np.nanmax(F22_diff)
F22im.set_clim(F22min, F22max)
ax3.set_title('F22')

plt.savefig('Deformation Differences.png')

# Statistics on difference in deformation

F11_diff_mean = np.nanmean(F11_diff)
F12_diff_mean = np.nanmean(F12_diff)
F21_diff_mean = np.nanmean(F21_diff)
F22_diff_mean = np.nanmean(F22_diff)

F11_SEM = stats.sem(F11_diff, axis=None)
F12_SEM = stats.sem(F12_diff, axis=None)
F21_SEM = stats.sem(F21_diff, axis=None)
F22_SEM = stats.sem(F22_diff, axis=None)

# Write statistics to files

f = open('DifferenceStatistics.txt', 'w')
# Write U and V Stats
f.write('Errors between synthetic and minimization results\n')
f.write('Displacement Errors (Mean +/- SEM\n')
f.write('{0} +/- {1}\n'.format(U_diff_mean, U_SEM))
f.write('{0} +/- {1}\n'.format(V_diff_mean, V_SEM))
f.write('Deformation Errors (Mean +/- SEM\n')
f.write('{0} +/- {1}\n'.format(F11_diff_mean, F11_SEM))
f.write('{0} +/- {1}\n'.format(F12_diff_mean, F12_SEM))
f.write('{0} +/- {1}\n'.format(F21_diff_mean, F21_SEM))
f.write('{0} +/- {1}\n'.format(F22_diff_mean, F22_SEM))
f.close()
