import numpy as np
import torch
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import plotly.graph_objs as go
import plotly.io as pio

from src.grid_functions import get_nearest_voxels
from src.ray_sampling import normalize_samples_for_indecies, sample_camera_rays_batched, compute_alpha_weighted_pixels


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

def visulize_3d_in_2d(grid_cells_data, transform_matrices, camera_angle_x, imgs, grid_indices, do_threshold,
                      transparency_threshold, number_of_rays, num_samples, device="cpu"):

    grid_cells = grid_cells_data["grid"].detach().to(device)
    grid_cells = grid_cells.clip(0.0, 1.0)
    if do_threshold:
        alphas = grid_cells[..., -1]
        grid_cells[..., -1][alphas < transparency_threshold] = 0.0

    gridsize = grid_cells.shape
    points_distance = grid_cells_data["param"]["points_distance"]
    delta_step = grid_cells_data["param"]["delta_step"]

    # generate samples
    samples_interval, pixels_to_rays, camera_positions, ray_directions = sample_camera_rays_batched(
        transform_matrices=transform_matrices,
        camera_angle_x=camera_angle_x,
        imgs=imgs,
        number_of_rays=number_of_rays,
        num_samples=num_samples,
        delta_step=delta_step,
        even_spread=True,
        camera_ray=False,
        device=device
    )

    # compute closest grid points to samples
    normalized_samples_for_indecies = normalize_samples_for_indecies(grid_indices, samples_interval, points_distance)

    # assign samples to voxels
    nearest, mask = get_nearest_voxels(normalized_samples_for_indecies, grid_cells)
    nearest = nearest * mask.unsqueeze(-1)  # zero the samples landing outside the grid

    # find pixels color
    nearest = nearest.reshape(1, number_of_rays, num_samples, 4)
    pixels_color = compute_alpha_weighted_pixels(nearest)
    res = int(np.sqrt(number_of_rays))

    # reshape and normalize an image and show
    pixels_color = pixels_color.cpu().detach().numpy()
    pixels_color = (pixels_color * 255).round().clip(0, 255).astype(np.uint8)
    image = pixels_color.reshape(res, res, 4)

    return np.transpose(image, (1, 0, 2))


def visulize_3d_in_2d_fast(grid, points_distance, transform_matrix, camera_angle_x, size_y):
    # Load camera parameters ---

    rotation_matrix = transform_matrix[:3, :3]
    camera_normal = -rotation_matrix[:, 2]
    pos = transform_matrix[:3, 3]

    camera_yz_plane_normal = transform_matrix[:3, 0]  # also the Y axis used for the camera
    camera_xz_plane_normal = transform_matrix[:3, 1]  # also the X axis used for the camera
    # camera_xy_plane_normal = transform_matrix[:3, 2] # unused

    aspect_ratio = camera_yz_plane_normal.norm(dim=0) / camera_xz_plane_normal.norm(dim=0)

    # get points ---

    condition = grid[..., 3] > 0.1
    points = np.argwhere(condition)
    points_colors = grid[condition]

    # normalize points to world coordinates ---

    grid_halfsize = (torch.tensor(grid.shape[:3]) / 2).unsqueeze(1).ceil()
    points = (points - grid_halfsize) + 1
    points = (points * points_distance).T

    # get location of points on screen via camera fov in x and y axis ---

    camera_to_points = points - pos

    yz_dot_products = torch.matmul(camera_to_points, camera_yz_plane_normal)
    xz_dot_products = torch.matmul(camera_to_points, camera_xz_plane_normal)

    norms_mul = (camera_to_points.norm(dim=1) * camera_yz_plane_normal.norm())
    radians_angles_x = yz_dot_products / norms_mul
    radians_angles_y = xz_dot_products / norms_mul

    x_locs_normalized = 0.5 + radians_angles_y / -camera_angle_x
    y_locs_normalized = 0.5 + radians_angles_x / (camera_angle_x / aspect_ratio)

    # remove points outside the fov ---

    unfitxy = (
            x_locs_normalized < 1.0
    ).logical_and(x_locs_normalized >= 0.0
                  ).logical_and(y_locs_normalized < 1.0
                                ).logical_and(y_locs_normalized >= 0.0
                                              )

    x_locs_normalized = x_locs_normalized[unfitxy]
    y_locs_normalized = y_locs_normalized[unfitxy]
    points_colors = points_colors[unfitxy]
    camera_to_points = camera_to_points[unfitxy]

    # normalize to image space ---

    ys = size_y
    xs = int(ys * aspect_ratio)

    xx = (xs * x_locs_normalized).round().clamp(min=0, max=xs - 1).type(torch.long)
    yy = (ys * y_locs_normalized).round().clamp(min=0, max=ys - 1).type(torch.long)

    # sort points via distance from camera ---

    distances = camera_to_points.norm(dim=1)
    sorted_indices = torch.argsort(-distances)
    xx, yy, points_colors = xx[sorted_indices], yy[sorted_indices], points_colors[sorted_indices]

    # draw image --

    img = np.ones([xs, ys, 3])
    img[xx, yy] = points_colors[:, :3]

    return img

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
