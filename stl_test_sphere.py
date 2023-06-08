import numpy as np
from skimage import measure
from stl import mesh
from scipy.ndimage import gaussian_filter

# Create a 3D volume with a sphere in the center
volume = np.zeros((100, 100, 100), dtype=np.float32)
x, y, z = np.indices((100, 100, 100))
sphere = (x - 50)**2 + (y - 50)**2 + (z - 50)**2
volume[np.where(sphere < 800)] = 1

# Apply marching cubes
verts, faces, normals, values = measure.marching_cubes(volume, level=0.5)

# Create mesh
my_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))

# Populate the mesh with data
for i, f in enumerate(faces):
    for j in range(3):
        my_mesh.vectors[i][j] = verts[f[j],:]

# Write the mesh to file "sphere.stl"
my_mesh.save('sphere.stl')

# Apply a Gaussian filter to the volume
volume = gaussian_filter(volume, sigma=3)

# Apply marching cubes
verts, faces, normals, values = measure.marching_cubes(volume, level=0.5)

# Create mesh
my_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))

# Populate the mesh with data
for i, f in enumerate(faces):
    for j in range(3):
        my_mesh.vectors[i][j] = verts[f[j],:]

# Write the mesh to file "smooth_sphere.stl"
my_mesh.save('smooth_sphere.stl')
