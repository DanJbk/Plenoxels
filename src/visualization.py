import numpy as np
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import plotly.graph_objs as go
import plotly.io as pio


def visulize_rays(ray_directions, camera_positions):
    # Orthographic projection: discard the z-coordinate
    ray_directions_2d = ray_directions[:, :2]

    # Plot the 2D vectors
    plt.figure(figsize=(10, 10))
    origin_x = camera_positions[:, 0]
    origin_y = camera_positions[:, 1]

    plt.quiver(origin_x, origin_y,
               ray_directions_2d[:, 0], ray_directions_2d[:, 1], angles='xy', scale_units='xy',
               scale=1)
    # plt.xlim(-2, 2)
    # plt.ylim(-2, 2)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.gca().set_aspect('equal')
    plt.title("Orthographic Projection of Evenly Spread Ray Directions within Camera's FOV")
    plt.grid()
    plt.show()


def visualize_rays_3d(ray_directions, camera_positions, red=None, green=None, orange=None):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    if len(camera_positions) > 0:
        origin_x = camera_positions[:, 0]
        origin_y = camera_positions[:, 1]
        origin_z = camera_positions[:, 2]

        # Plot 3D vectors
        for i in (range(ray_directions.shape[0])):
            ax.quiver(origin_x[i], origin_y[i], origin_z[i],
                      ray_directions[i, 0], ray_directions[i, 1], ray_directions[i, 2],
                      color='b', alpha=0.5)

    for item, color in zip([red, green, orange], ['r', 'g', 'orange']):
        if not item is None:
            scatter_x = item[:, 0]
            scatter_y = item[:, 1]
            scatter_z = item[:, 2]
            ax.scatter(scatter_x, scatter_y, scatter_z, c=color, marker='o')

    ax.set_xlim3d(-5, 5)
    ax.set_ylim3d(-5, 5)
    ax.set_zlim3d(-2, 5 * 2)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title("3D Visualization of Evenly Spread Ray Directions within Camera's FOV")
    plt.show()


def visualize_rays_3d_plotly(ray_directions, camera_positions, red=None, green=None, orange=None):
    fig = go.Figure()

    if len(camera_positions) > 0:
        origin_x = camera_positions[:, 0]
        origin_y = camera_positions[:, 1]
        origin_z = camera_positions[:, 2]

        for i in range(ray_directions.shape[0]):
            color = 'blue' if (i - 1) % 25 != 0 else 'red'
            alpha = 0.5

            x = np.array([origin_x[i], origin_x[i] + ray_directions[i, 0]])
            y = np.array([origin_y[i], origin_y[i] + ray_directions[i, 1]])
            z = np.array([origin_z[i], origin_z[i] + ray_directions[i, 2]])

            fig.add_trace(go.Scatter3d(x=x, y=y, z=z,
                                       marker=dict(size=0),
                                       line=dict(
                                           color=f'rgba({",".join(map(str, mcolors.to_rgba(color)[:3]))}, {alpha})'),
                                       showlegend=False))

    for item, color in zip([red, green, orange], ['red', 'green', 'orange']):
        if item is not None:
            scatter_x = item[:, 0]
            scatter_y = item[:, 1]
            scatter_z = item[:, 2]

            fig.add_trace(go.Scatter3d(x=scatter_x, y=scatter_y, z=scatter_z,
                                       mode='markers',
                                       marker=dict(size=5, color=color),
                                       showlegend=False))

    fig.update_layout(scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z',
                                 aspectmode='cube',
                                 xaxis=dict(range=[-5, 5]),
                                 yaxis=dict(range=[-5, 5]),
                                 zaxis=dict(range=[-2, 10])),
                      title="3D Visualization of Evenly Spread Ray Directions within Camera's FOV",
                      autosize=False, width=800, height=800)
    pio.write_html(fig, file='visualize_rays_3d.html', auto_open=True)

    # fig.show()


def voxel_visulization(index, grid_grid, num_samples, number_of_rays, selected_points_voxels, samples_interval,
                       ray_directions):
    index0 = index * num_samples * number_of_rays
    index1 = index * num_samples * number_of_rays + num_samples * number_of_rays
    temp = selected_points_voxels[index0: index1]
    temp = grid_grid[temp[:, 0], temp[:, 1], temp[:, 2]]
    # choose samples along a ray
    temp2 = samples_interval[
            index * num_samples * number_of_rays: index * num_samples * number_of_rays + num_samples * number_of_rays]
    visualize_rays_3d_plotly(ray_directions, [], temp, temp2)


def view_opacity_histogram(grid_cells):
    np_array = grid_cells[..., -1].detach().flatten().cpu().numpy()
    plt.hist(np_array, bins=900)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Histogram of PyTorch Tensor Values')

    # Display the histogram
    plt.show()


def paper_visulization(index, grid_grid, num_samples, number_of_rays, selected_points, samples_interval,
                       ray_directions):
    index0 = index * num_samples * number_of_rays
    index1 = index * num_samples * number_of_rays + num_samples * number_of_rays
    temp = selected_points[index0: index1].reshape([number_of_rays * num_samples * 8, 3])
    temp = grid_grid[temp[:, 0], temp[:, 1], temp[:, 2]]

    # choose samples along a ray
    temp2 = samples_interval[
            index * num_samples * number_of_rays: index * num_samples * number_of_rays + num_samples * number_of_rays]
    visualize_rays_3d_plotly(ray_directions, [], temp, temp2)
