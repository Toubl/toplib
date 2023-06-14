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

def add_buffer(volume, buffer_size):
    buffer = np.zeros((volume.shape[0] + 2*buffer_size,
                       volume.shape[1] + 2*buffer_size,
                       volume.shape[2] + 2*buffer_size))
    buffer[buffer_size:-buffer_size, buffer_size:-buffer_size, buffer_size:-buffer_size] = volume
    return buffer

voxel_grid = np.loadtxt('x_opt_40_15_15.txt')
volume = voxel_grid.reshape((40, 15, 15))
print(volume.min(), volume.max())

# Refine the voxel grid
refined_volume = refine_grid(volume, 10)

# Add a buffer around the refined volume
buffered_volume = add_buffer(refined_volume, buffer_size=4)  # adjust the buffer size as needed

# Apply marching cubes to the refined volume
verts, faces, normals, values = measure.marching_cubes(refined_volume, level=0.5)
save_as_stl(verts, faces, 'STL_files/initial.stl')  # Save the initial object as STL

# Apply a Gaussian filter to the buffered volume
volume_smooth = gaussian_filter(buffered_volume, sigma=6.1)
print(volume_smooth.min(), volume_smooth.max())

# Apply marching cubes to the smoothed volume
verts_smooth, faces_smooth, normals_smooth, values_smooth = measure.marching_cubes(volume_smooth, level=0.5)
save_as_stl(verts_smooth, faces_smooth, 'STL_files/outer_shell_added_large.stl')  # Save the smoothed object as STL
