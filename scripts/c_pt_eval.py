import numpy as np
import nurbs
from splineDIC import knots
from splineDIC import spline

deg = 3
ncpts = 6
kv = knots.generate_uniform(deg, ncpts)

cpts = np.linspace(0.0, 4.0, ncpts)

ys, xs = np.meshgrid(cpts, cpts)
xcpts = xs.flatten()
ycpts = ys.flatten()

u = np.linspace(0, 1, 800)
uu, vv = np.meshgrid(u, u)

us = uu.flatten()
vs = vv.flatten()

uv = np.column_stack((us, vs))

surf = spline.Surface()

surf.degree_u = deg
surf.degree_v = deg

surf.num_ctrlpts_u = ncpts
surf.num_ctrlpts_v = ncpts

surf.control_points = np.column_stack((xcpts, ycpts, np.zeros(len(xcpts))))

surf.knot_vector_u = kv
surf.knot_vector_v = kv

foo = surf.points(uv)
