
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



def draw(npy_path, use_hi=True):
    # Load the .npy file
    point_cloud = np.load(npy_path)

    # Separate coordinates and colors
    x, y, z = point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2]
    if not use_hi:
        r, g, b = point_cloud[:, 3], point_cloud[:, 4], point_cloud[:, 5]
    else:
        r, g, b = point_cloud[:, 6], point_cloud[:, 7], point_cloud[:, 8]


    # Normalize the color values from 0-255 to 0-1 range
    r, g, b = r / 255.0, g / 255.0, b / 255.0

    # Create a new figure and a 3D subplot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Equal aspect ratio for all axes to avoid distortion
    ax.set_box_aspect([np.ptp(i) for i in [x, y, z]])

    # Plot the point cloud
    scatter = ax.scatter(x, y, z, c=[(r, g, b) for r, g, b in zip(r, g, b)], s=1)

    # Set the title and labels for the axes
    ax.set_title('3D Point Cloud Visualization')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_zlabel('Z Coordinate')

    # Add a color bar
    cbar = fig.colorbar(scatter, ax=ax, pad=0.2)
    cbar.set_label('RGB Color')

    # Display the plot
    # plt.show()


num = 13

npy_file_path_hi = f'H:\CodeUndergrad\Refer3dProject\Transcrib3D\data\pointcloud_label_data_hi-res\scene0000_00\scene0000_00_{num}_pointcloud.npy'  # Replace with the path to your .npy file

npy_file_path_lo = f"H:\CodeUndergrad\Refer3dProject\pointcloud_label_data\scene0000_00\scene0000_00_{num}_pointcloud.npy"

draw(npy_file_path_lo, use_hi=False)
draw(npy_file_path_hi)

plt.show()
