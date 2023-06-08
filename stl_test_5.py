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

def apply_inner_gaussian_filter(volume, sigma, shell_thickness):
    # Create a mask for the outer shell
    outer_shell_mask = np.ones(volume.shape, dtype=bool)
    outer_shell_mask[shell_thickness:-shell_thickness, shell_thickness:-shell_thickness, shell_thickness:-shell_thickness] = False

    # Create a mask for the inner part
    inner_mask = np.logical_not(outer_shell_mask)

    # Extract the inner part and outer shell
    inner_volume = volume * inner_mask
    outer_shell = volume * outer_shell_mask

    # Apply the Gaussian filter to the inner part
    inner_volume_smooth = gaussian_filter(inner_volume, sigma)
    
    # Zero out the outer shell of the smoothed inner volume
    inner_volume_smooth = inner_volume_smooth * inner_mask

    # Combine the smoothed inner part and the outer shell
    combined_volume = inner_volume_smooth + outer_shell

    return combined_volume, outer_shell

# Load and reshape the voxel grid
voxel_grid = np.loadtxt('x_opt_20_10_10.txt')
volume = voxel_grid.reshape((20, 10, 10))

# Refine the voxel grid
refined_volume = refine_grid(volume, 6)

# Add a buffer around the refined volume
buffered_volume = add_buffer(refined_volume, buffer_size=3)  # adjust the buffer size as needed

# Apply marching cubes to the refined volume
verts, faces, normals, values = measure.marching_cubes(refined_volume, level=0.5)
save_as_stl(verts, faces, 'STL_files/initial.stl')  # Save the initial object as STL

# Apply a Gaussian filter to the inner part of the buffered volume
volume_smooth, outer_shell = apply_inner_gaussian_filter(buffered_volume, sigma=2.2, shell_thickness=4)

# Apply final Gaussian filter to the combined volume
sigma_final = 1.0  # You can adjust this value
final_volume_smooth = gaussian_filter(volume_smooth, sigma_final)

# Apply marching cubes to the final smoothed volume
verts_smooth, faces_smooth, normals_smooth, values_smooth = measure.marching_cubes(final_volume_smooth, level=0.5)
save_as_stl(verts_smooth, faces_smooth, 'STL_files/smooth_final.stl')  # Save the final smoothed
