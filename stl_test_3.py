import numpy as np
from skimage import measure
from scipy.ndimage import gaussian_filter
from stl import mesh

# Function to save STL
def save_as_stl(verts, faces, filename):
    my_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            my_mesh.vectors[i][j] = verts[f[j],:]
    my_mesh.save(filename)

# Create an empty 3D volume
volume = np.zeros((200, 200, 200), dtype=np.float32)
x, y, z = np.indices((200, 200, 200))

# Create a sphere in the center
sphere = (x - 100)**2 + (y - 100)**2 + (z - 100)**2
volume[np.where(sphere < 800)] = 1

# Create a cube on the right
cube = (x > 130) & (x < 170) & (y > 130) & (y < 170) & (z > 130) & (z < 170)
volume[np.where(cube)] = 1

# Create a pyramid on the left
pyramid = 50 - abs(x - 30) - abs(y - 130) - abs(z - 130)
volume[np.where(pyramid > 0)] = 1

# Apply marching cubes
verts, faces, normals, values = measure.marching_cubes(volume, level=0.5)

# Save the combined object as STL
save_as_stl(verts, faces, 'combined.stl')

# Apply a Gaussian filter to the volume
volume_smooth = gaussian_filter(volume, sigma=2)

# Apply marching cubes to the smoothed volume
verts_smooth, faces_smooth, normals_smooth, values_smooth = measure.marching_cubes(volume_smooth, level=0.5)

# Save the smoothed combined object as STL
save_as_stl(verts_smooth, faces_smooth, 'smooth_combined.stl')
