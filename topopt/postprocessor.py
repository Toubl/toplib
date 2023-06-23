import numpy as np
from skimage import measure
from scipy.ndimage import gaussian_filter
from stl import mesh


class Postprocessor:
    """Postprocessor for topology optimization results."""

    def __init__(self, volume):
        """
        Initialize the Postprocessor with a voxel grid.

        Parameters
        ----------
        volume : ndarray
            3D array of the voxel grid.
        """
        self.volume = volume
        
    def load_and_reshape_grid(self, filepath, shape=(20, 10, 10)):
        """
        Load and reshape the voxel grid from a text file.

        Parameters
        ----------
        filepath : str
            Path to the voxel grid file.
        shape : tuple of int, optional
            Shape of the reshaped voxel grid (default is (20, 10, 10)).

        Returns
        -------
        volume : ndarray
            3D array of the reshaped voxel grid.
        """
        voxel_grid = np.loadtxt(filepath)
        return voxel_grid.reshape(shape)

    def save_as_stl(self, verts, faces, filename):
        """
        Save the voxel grid as an STL file.

        Parameters
        ----------
        verts : ndarray
            Vertices of the voxel grid.
        faces : ndarray
            Faces of the voxel grid.
        filename : str
            Path to the output STL file.
        """
        my_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
        for i, f in enumerate(faces):
            for j in range(3):
                my_mesh.vectors[i][j] = verts[f[j], :]
        my_mesh.save(filename)

    def refine_grid(self, volume, factor):
        """
        Refine the voxel grid.

        Parameters
        ----------
        volume : ndarray
            3D array of the voxel grid.
        factor : int
            Factor by which to refine the grid.

        Returns
        -------
        new_volume : ndarray
            3D array of the refined voxel grid.
        """
        x, y, z = volume.shape
        new_volume = np.repeat(np.repeat(np.repeat(volume, factor, axis=0), factor, axis=1), factor, axis=2)
        return new_volume.reshape(x * factor, y * factor, z * factor)

    def add_buffer(self, volume, buffer_size):
        """
        Add a buffer around the voxel grid.

        Parameters
        ----------
        volume : ndarray
            3D array of the voxel grid.
        buffer_size : int
            Size of the buffer to add.

        Returns
        -------
        buffer : ndarray
            3D array of the voxel grid with a buffer.
        """
        buffer = np.zeros((volume.shape[0] + 2*buffer_size,
                           volume.shape[1] + 2*buffer_size,
                           volume.shape[2] + 2*buffer_size))
        buffer[buffer_size:-buffer_size, buffer_size:-buffer_size, buffer_size:-buffer_size] = volume
        return buffer

    def apply_inner_gaussian_filter(self, volume, sigma, shell_thickness):
        """
        Apply a Gaussian filter to the inner part of the voxel grid.

        Parameters
        ----------
        volume : ndarray
            3D array of the voxel grid.
        sigma : float
            Standard deviation for the Gaussian kernel.
        shell_thickness : int
            Thickness of the shell to exclude from the filtering.

        Returns
        -------
        combined_volume : ndarray
            3D array of the voxel grid with the filtered inner part.
        outer_shell : ndarray
            3D array of the outer shell of the voxel grid.
        """
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

    def process(self, refinement_factor, buffer_size, inner_gaussian_sigma, shell_thickness, final_gaussian_sigma, initial_filename, final_filename):
        """
        Process the voxel grid.

        Parameters
        ----------
        refinement_factor : int
            Factor by which to refine the grid.
        buffer_size : int
            Size of the buffer to add.
        inner_gaussian_sigma : float
            Standard deviation for the Gaussian filter for the inner part.
        shell_thickness : int
            Thickness of the shell to exclude from the filtering.
        final_gaussian_sigma : float
            Standard deviation for the final Gaussian filter.
        initial_filename : str
            Path to the output STL file for the initial object.
        final_filename : str
            Path to the output STL file for the final smoothed volume.

        """
        # Refine the voxel grid
        refined_volume = self.refine_grid(self.volume, refinement_factor)

        # Add a buffer around the refined volume
        buffered_volume = self.add_buffer(refined_volume, buffer_size)

        # Set the threshold level to be the midpoint of min and max of volume data
        level = (np.min(refined_volume) + np.max(refined_volume)) / 2

        # Apply marching cubes to the refined volume
        verts, faces, normals, values = measure.marching_cubes(refined_volume, level=level)
        self.save_as_stl(verts, faces, initial_filename)  # Save the initial object as STL

        # Apply a Gaussian filter to the inner part of the buffered volume
        volume_smooth, outer_shell = self.apply_inner_gaussian_filter(buffered_volume, inner_gaussian_sigma, shell_thickness)

        # Apply final Gaussian filter to the combined volume
        final_volume_smooth = gaussian_filter(volume_smooth, final_gaussian_sigma)

        # Set the threshold level to be the midpoint of min and max of smoothed volume data
        level_smooth = (np.min(final_volume_smooth) + np.max(final_volume_smooth)) / 2

        # Apply marching cubes to the final smoothed volume
        verts_smooth, faces_smooth, normals_smooth, values_smooth = measure.marching_cubes(final_volume_smooth, level=level_smooth)
        self.save_as_stl(verts_smooth, faces_smooth, final_filename)  # Save the final smoothed volume
