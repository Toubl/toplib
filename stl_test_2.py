import numpy as np
from skimage import measure
from scipy.ndimage import gaussian_filter
from stl import mesh

def save_as_stl(verts, faces, filename):
    my_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            my_mesh.vectors[i][j] = verts[f[j],:]
    my_mesh.save(filename)

def refine_grid(volume, factor):
    x, y, z = volume.shape
    new_volume = np.repeat(np.repeat(np.repeat(volume, factor, axis=0), factor, axis=1), factor, axis=2)
    return new_volume.reshape(x * factor, y * factor, z * factor)

voxel_grid = np.loadtxt('x_opt_20_10_10.txt')
volume = voxel_grid.reshape((20, 10, 10))
print(volume.min(), volume.max())

# Refine the voxel grid
refined_volume = refine_grid(volume, 6)
print(refined_volume.min(), refined_volume.max())

# Apply marching cubes to the refined volume
verts, faces, normals, values = measure.marching_cubes(refined_volume, level=0.5)
save_as_stl(verts, faces, 'STL_files/initial.stl')  # Save the initial object as STL

# Apply a Gaussian filter to the refined volume
volume_smooth = gaussian_filter(refined_volume, sigma=2)
print(volume_smooth.min(), volume_smooth.max())

# Apply marching cubes to the smoothed volume
verts_smooth, faces_smooth, normals_smooth, values_smooth = measure.marching_cubes(volume_smooth, level=0.5)
save_as_stl(verts_smooth, faces_smooth, 'STL_files/smooth.stl')  # Save the smoothed object as STL
