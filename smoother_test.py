import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

def smooth(voxel_data):
    # Apply the Gaussian filter
    smoothed_voxel_data = gaussian_filter(voxel_data, sigma=7)

    # Choose a slice index that is halfway along the first axis
    slice_idx = voxel_data.shape[0] // 2

    # Plot a slice from the original voxel data
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.title('Original Voxel Data')
    plt.imshow(voxel_data, cmap='gray')
    plt.colorbar()

    # Plot a slice from the smoothed voxel data
    plt.subplot(1, 2, 2)
    plt.title('Smoothed Voxel Data')
    plt.imshow(smoothed_voxel_data, cmap='gray')
    plt.colorbar()

    plt.show()

# Assume voxel_data is your 3D numpy array.
voxel_data = np.random.rand(100,100)  # replace this with your actual voxel data
smooth(voxel_data)


