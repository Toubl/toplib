import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from skimage import measure
from stl import Mesh
from mayavi import mlab

# Assume 'voxels' is your 3D numpy array representing voxel data
voxels = np.random.rand(40,30,50)  # replace with your actual data

print(voxels)
# Show original voxel data
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.voxels(voxels, edgecolor='k')
# plt.show()

# Apply the Gaussian filter for smoothing
smoothed_voxels = gaussian_filter(voxels, sigma=2)

# Apply marching cubes
verts, faces, normals, values = measure.marching_cubes(smoothed_voxels)

# Create the mesh
voxel_mesh = Mesh(np.zeros(faces.shape[0], dtype=Mesh.dtype))
for i, f in enumerate(faces):
    for j in range(3):
        voxel_mesh.vectors[i][j] = verts[f[j],:]

# Save to file
voxel_mesh.save('STL_files/voxel_mesh.stl')

# Visualize the resulting mesh
mlab.triangular_mesh([vert[0] for vert in verts],
                     [vert[1] for vert in verts],
                     [vert[2] for vert in verts],
                     faces)
mlab.show()