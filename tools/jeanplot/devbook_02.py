import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.patches as mpatches


N = 90
vertices = np.random.rand(N, 2) * 1000
##
codes = [mpath.Path.MOVETO] + [mpath.Path.LINETO] * (N - 1)
path = mpath.Path(vertices, codes)
fig, axes = plt.subplots(2,1,figsize=(5, 10), dpi=200)
ax = axes[0]
ax.set_aspect('equal')
patch = mpatches.PathPatch(path, fc='none', ec='k', lw=1)
ax.plot(vertices[:, 0], vertices[:, 1], 'x', ms=3, color='b')
# plot first as green and last as red
ax.plot(vertices[0, 0], vertices[0, 1], 'o', ms=5, color='g')
ax.plot(vertices[-1, 0], vertices[-1, 1], 'o', ms=5, color='r')
ax.set_ylim(0, 1000)
ax.set_xlim(0, 1000)
ax.add_patch(patch)


corner_indices = np.arange(0, vertices.shape[0] - 2)[:, None] + np.array([0, 1, 2])
corner_indices
corners = vertices[corner_indices]
corn = corners[0]

rounded_vertices = [vertices[0]]
rounded_codes = [codes[0]]
radius = 1000

for corn in corners:
    a, b, c = corn
    ba, bc = a - b, c - b
    ba_len, bc_len = np.linalg.norm(ba), np.linalg.norm(bc)
    ba_unit, bc_unit = ba / np.maximum(ba_len, 1e-8), bc / np.maximum(bc_len, 1e-8)
    p0 = b + ba_unit * np.minimum(radius, ba_len / 2)
    p1 = b + bc_unit * np.minimum(radius, bc_len / 2)
    p0b = p0 + (b - p0) / 2
    p1b = p1 + (b - p1) / 2
    rounded_vertices += [p0, p0b, p1b, p1]
    rounded_codes += [mpath.Path.LINETO, mpath.Path.CURVE4, mpath.Path.CURVE4, mpath.Path.CURVE4]

rounded_vertices += [vertices[-1]]
rounded_codes += [codes[-1]]
rounded_vertices = np.array(rounded_vertices)
ax = axes[1]
ax.set_aspect('equal')
rounded_path = mpath.Path(rounded_vertices, rounded_codes)
patch = mpatches.PathPatch(rounded_path, fc='none', ec='k', lw=1)
ax.add_patch(patch)
ax.set_ylim(0, 1000)
ax.set_xlim(0, 1000)
# ax.plot(rounded_vertices[:, 0], rounded_vertices[:, 1], 'x', ms=3, color='b')

# as a function
def make_round_path(vertices, radius):
    corner_indices = np.arange(0, vertices.shape[0] - 2)[:, None] + np.array([0, 1, 2])
    corners = vertices[corner_indices]
    rounded_vertices = [vertices[0]]
    rounded_codes = [codes[0]]

    for corn in corners:
        a, b, c = corn
        ba, bc = a - b, c - b
        ba_len, bc_len = np.linalg.norm(ba), np.linalg.norm(bc)
        ba_unit, bc_unit = ba / np.maximum(ba_len, 1e-8), bc / np.maximum(bc_len, 1e-8)
        p0 = b + ba_unit * np.minimum(radius, ba_len / 2)
        p1 = b + bc_unit * np.minimum(radius, bc_len / 2)
        p0b = p0 + (b - p0) / 2
        p1b = p1 + (b - p1) / 2
        rounded_vertices += [p0, p0b, p1b, p1]
        rounded_codes += [mpath.Path.LINETO, mpath.Path.CURVE4, mpath.Path.CURVE4, mpath.Path.CURVE4]

    rounded_vertices += [vertices[-1]]
    rounded_codes += [codes[-1]]
    rounded_vertices = np.array(rounded_vertices)
    rounded_path = mpath.Path(rounded_vertices, rounded_codes)
    return rounded_path

