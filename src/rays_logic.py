import json
import time
import torch
import logging
import numpy as np
import matplotlib.pyplot as plt
from logging import info as printi
from PIL import Image

import plotly.graph_objs as go
import plotly.io as pio
import matplotlib.colors as mcolors

from mpl_toolkits.mplot3d import Axes3D

log_color = "\x1b[32;1m"
logging.basicConfig(level=logging.INFO, format=f'{log_color}%(asctime)s - %(message)s')


def load_data(data):
    transform_matricies = []
    file_paths = []
    for i in range(len(data["frames"])):
        transform_matrix, rotation, file_path, camera_angle_x = get_data_from_index(data, i)
        transform_matricies.append(transform_matrix.unsqueeze(0))
        file_paths.append(file_path)

    transform_matricies = torch.cat(transform_matricies, 0)

    return transform_matricies, file_paths, camera_angle_x


def sample_camera_rays_batched(transform_matricies, camera_angle_x, imgs, number_of_rays, num_samples, delta_step,
                               even_spread, camera_ray):
    if even_spread:
        number_of_rays = int(np.round(np.sqrt(number_of_rays)) ** 2)

    # transform_matricies, file_paths, camera_angle_x = load_data(data)

    pixels_to_rays = []
    if camera_ray:
        current_ray_directions = transform_matricies[:, :3, 2].unsqueeze(0) * -1

    else:
        current_ray_directions, pixels_to_rays = generate_rays_batched(imgs, number_of_rays, transform_matricies,
                                                                       camera_angle_x,
                                                                       even_spread=even_spread)

    ray_directions = current_ray_directions
    camera_positions = transform_matricies[:, :3, 3]

    delta_forsamples = delta_step * torch.arange(num_samples + 1)[1:].repeat(number_of_rays * imgs.shape[0]) \
        .unsqueeze(1)

    camera_positions_forsamples = torch.repeat_interleave(camera_positions, num_samples * number_of_rays, 0)

    ray_directions_for_samples = torch.repeat_interleave(ray_directions, num_samples, 0)
    samples_interval = camera_positions_forsamples + ray_directions_for_samples * delta_forsamples

    return samples_interval, pixels_to_rays, camera_positions, ray_directions


def batched_cartesian_prod(A, B):
    A_expanded = A.unsqueeze(-1).expand(-1, -1, B.size(1))
    B_expanded = B.unsqueeze(-2).expand(-1, A.size(1), -1)
    return torch.stack((A_expanded, B_expanded), dim=-1).view(A.size(0), -1, 2)


def tensor_linspace(start, end, steps=10):
    """
    # https://github.com/zhaobozb/layout2im/blob/master/models/bilinear.py#L246

    Vectorized version of torch.linspace.
    Inputs:
    - start: Tensor of any shape
    - end: Tensor of the same shape as start
    - steps: Integer
    Returns:
    - out: Tensor of shape start.size() + (steps,), such that
      out.select(-1, 0) == start, out.select(-1, -1) == end,
      and the other elements of out linearly interpolate between
      start and end.
    """
    assert start.size() == end.size()
    view_size = start.size() + (1,)
    w_size = (1,) * start.dim() + (steps,)
    out_size = start.size() + (steps,)

    start_w = torch.linspace(1, 0, steps=steps).to(start)
    start_w = start_w.view(w_size).expand(out_size)
    end_w = torch.linspace(0, 1, steps=steps).to(start)
    end_w = end_w.view(w_size).expand(out_size)

    start = start.contiguous().view(view_size).expand(out_size)
    end = end.contiguous().view(view_size).expand(out_size)

    out = start_w * start + end_w * end
    return out


def generate_rays_batched(imgs, number_of_rays, transform_matricies, camera_angle_x, even_spread=False):
    """
        Generates rays for each camera and corresponding pixel indices.

        Args:
            imgs (torch.Tensor): Batch of input images (B, H, W).
            number_of_rays (int): Number of rays to generate for each camera.
            transform_matricies (torch.Tensor): Transformation matrices for cameras (B, 4, 4).
            camera_angle_x (float): Horizontal field of view of the camera.
            even_spread (bool, optional): Whether to generate rays with even spacing. Defaults to False (random spacing).

        Returns:
            torch.Tensor: Generated ray directions (B * number_of_rays, 3).
            torch.Tensor: Pixel colors corresponding to the generated rays.
        """

    num_cameras = transform_matricies.shape[0]

    # Extract camera axes from transformation matrices
    camera_x_axis = transform_matricies[:, :3, 0]
    camera_y_axis = transform_matricies[:, :3, 1]
    camera_z_axis = -transform_matricies[:, :3, 2]
    aspect_ratio = camera_x_axis.norm(dim=1) / camera_y_axis.norm(dim=1)

    if even_spread:
        num_rays_sqrt = np.round(np.sqrt(number_of_rays))

        # Compute evenly spaced u and v values for rays
        start = torch.tensor([-0.5 * camera_angle_x]).repeat(aspect_ratio.shape)
        u_values = torch.linspace(-0.5 * camera_angle_x, 0.5 * camera_angle_x, int(num_rays_sqrt)).unsqueeze(0).repeat(
            [aspect_ratio.shape[0], 1])
        v_values = -tensor_linspace(start / aspect_ratio, -start / aspect_ratio, int(num_rays_sqrt))

        # Calculate ray indices using Cartesian product
        ray_indices = batched_cartesian_prod(u_values, v_values)

    else:
        # Generate random ray indices
        ray_indices = torch.rand(transform_matricies.shape[0], number_of_rays,
                                 2)  # Generate random numbers in the range [0, 1)

        # Scale and shift the ray indices to match the camera's FOV
        ray_indices[:, :, 0] = camera_angle_x * (ray_indices[:, :, 0] - 0.5)
        ray_indices[:, :, 1] = -((camera_angle_x * (1 / aspect_ratio)).unsqueeze(1) * (ray_indices[:, :, 1] - 0.5))

    # normalize to image space
    pixel_indices = ray_indices.clone()
    # print(f"{ray_indices.shape=}\n{ray_indices}\n{ray_indices.reshape(3,3,2)}")
    pixel_indices[:, :, 0] = (pixel_indices[:, :, 0] / camera_angle_x) + 0.5
    pixel_indices[:, :, 1] = -(pixel_indices[:, :, 1] / ((camera_angle_x * (1 / aspect_ratio)).unsqueeze(1))) + 0.5

    # Clamp pixel indices to image dimensions
    pixel_indices[:, :, 0] = (imgs.shape[1] * pixel_indices[:, :, 0]).round().clamp(max=imgs.shape[1] - 1)
    pixel_indices[:, :, 1] = (imgs.shape[2] * pixel_indices[:, :, 1]).round().clamp(max=imgs.shape[2] - 1)
    pixel_indices = pixel_indices.to(torch.long)

    # Map pixel indices to camera indices
    camera_to_ray = torch.repeat_interleave(torch.arange(0, transform_matricies.shape[0]), number_of_rays, 0)
    pixel_indices = pixel_indices.reshape([pixel_indices.shape[0] * pixel_indices.shape[1], pixel_indices.shape[2]])
    # pixels_to_rays = imgs[camera_to_ray, pixel_indices[:, 0], pixel_indices[:, 1]]
    # the y dimension is the first one, the x dimension is the second in an image
    pixels_to_rays = imgs[camera_to_ray, pixel_indices[:, 1], pixel_indices[:, 0]]

    # Get the u and v values from ray_indices
    u_values, v_values = ray_indices[:, :, 0].unsqueeze(-1), ray_indices[:, :, 1].unsqueeze(-1)
    u_values = u_values.expand(num_cameras, number_of_rays, 3)
    v_values = v_values.expand(num_cameras, number_of_rays, 3)

    # Expand camera axes for broadcasting
    camera_x_axis = camera_x_axis.unsqueeze(1).expand(num_cameras, number_of_rays, 3)
    camera_y_axis = camera_y_axis.unsqueeze(1).expand(num_cameras, number_of_rays, 3)
    camera_z_axis = camera_z_axis.unsqueeze(1).expand(num_cameras, number_of_rays, 3)

    # Compute ray directions and normalize
    directions = u_values * camera_x_axis + v_values * camera_y_axis + camera_z_axis
    ray_directions = directions / directions.norm(dim=2).unsqueeze(-1)

    return ray_directions.reshape(ray_directions.shape[0] * number_of_rays, -1), pixels_to_rays


def get_grid_points_indices(normalized_samples_for_indecies):
    """
    Given grid indices, a samples interval, and a points distance, this function calculates the indices of
    grid points surrounding each input point. It returns the indices of the 8 corners of the grid cell
    that encloses each input point.

    Args:
            :param normalized_samples_for_indecies:
    Returns:
        torch.Tensor: A tensor of shape (N, 8, 3) containing the indices of the 8 corners of the grid cell
                      that encloses each input point.
    """

    ceil_x = torch.ceil(normalized_samples_for_indecies[:, 0]).unsqueeze(1)
    ceil_y = torch.ceil(normalized_samples_for_indecies[:, 1]).unsqueeze(1)
    ceil_z = torch.ceil(normalized_samples_for_indecies[:, 2]).unsqueeze(1)
    floor_x = torch.floor(normalized_samples_for_indecies[:, 0]).unsqueeze(1)
    floor_y = torch.floor(normalized_samples_for_indecies[:, 1]).unsqueeze(1)
    floor_z = torch.floor(normalized_samples_for_indecies[:, 2]).unsqueeze(1)

    # generate the indecies of all the corners.
    relevant_points = [
        torch.cat([x_component, y_component, z_component], 1).unsqueeze(1)
        for x_component in [ceil_x, floor_x]
        for y_component in [ceil_y, floor_y]
        for z_component in [ceil_z, floor_z]
    ]
    relevant_points = torch.cat(relevant_points, 1).type(torch.long)

    return relevant_points



def get_grid(sx, sy, sz, points_distance=0.5, info_size=4):
    grindx_indices, grindy_indices, grindz_indices = torch.arange(sx), torch.arange(sy), torch.arange(sz)
    coordsx, coordsy, coordsz = torch.meshgrid(grindx_indices, grindy_indices, grindz_indices, indexing='ij')

    meshgrid = torch.stack([coordsx, coordsy, coordsz], dim=-1)

    # center grid
    coordsx, coordsy, coordsz = coordsx - np.ceil(sx / 2) + 1, coordsy - np.ceil(sy / 2) + 1, coordsz - np.ceil(
        sz / 2) + 1

    # edit grid spacing
    coordsx, coordsy, coordsz = coordsx * points_distance, coordsy * points_distance, coordsz * points_distance

    # make it so no points of the grid are underground
    coordsz = coordsz - coordsz.min()

    grid_grid = torch.stack([coordsx, coordsy, coordsz], dim=-1)
    grid_coords = grid_grid.reshape(sx * sy * sz, 3)

    grid_cells = torch.zeros([grid_grid.shape[0], grid_grid.shape[1], grid_grid.shape[2], info_size],
                             requires_grad=True)

    return grid_coords, grid_cells, meshgrid, grid_grid


def get_data_from_index(data, index):
    camera_angle_x = data["camera_angle_x"]
    frame = data["frames"][index]

    file_path = frame["file_path"]
    rotation = frame["rotation"]
    transform_matrix = torch.tensor(frame["transform_matrix"])

    return transform_matrix, rotation, file_path, camera_angle_x


def generate_rays(num_rays, transform_matrix, camera_angle_x, even_spread=False):
    # Extract camera axes and position
    camera_x_axis = transform_matrix[:3, 0]
    camera_y_axis = transform_matrix[:3, 1]
    camera_z_axis = -transform_matrix[:3, 2]

    # Compute the aspect ratio (width / height) of the camera's FOV
    aspect_ratio = camera_x_axis.norm() / camera_y_axis.norm()

    # Generate evenly spread ray indices
    if even_spread:
        num_rays_sqrt = np.round(np.sqrt(num_rays))

        u_values = torch.linspace(-0.5 * camera_angle_x, 0.5 * camera_angle_x, int(num_rays_sqrt))
        v_values = torch.linspace(-0.5 * camera_angle_x / aspect_ratio, 0.5 * camera_angle_x / aspect_ratio,
                                  int(num_rays_sqrt))
        ray_indices = torch.cartesian_prod(u_values, v_values)

    else:
        # Generate random ray indices
        ray_indices = torch.rand(num_rays, 2)  # Generate random numbers in the range [0, 1)

        # Scale and shift the ray indices to match the camera's FOV
        ray_indices[:, 0] = camera_angle_x * (ray_indices[:, 0] - 0.5)
        ray_indices[:, 1] = camera_angle_x / aspect_ratio * (ray_indices[:, 1] - 0.5)

    # Get the u and v values from ray_indices
    u_values, v_values = ray_indices[:, 0].unsqueeze(-1), ray_indices[:, 1].unsqueeze(-1)
    directions = u_values * camera_x_axis + v_values * camera_y_axis + camera_z_axis

    ray_directions = directions / directions.norm(dim=1).unsqueeze(-1)

    return ray_directions


def get_camera_normal(transform_matrix):
    fixed_pos = transform_matrix[:3, :3].unsqueeze(0)
    fixed_pos[:, 2] = fixed_pos[:, 2] * -1
    return fixed_pos


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
            if (i - 8) % 9 == 0:

                ax.quiver(origin_x[i], origin_y[i], origin_z[i],
                          ray_directions[i, 0], ray_directions[i, 1], ray_directions[i, 2],
                          color='r', alpha=0.5)
            else:
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


def read_data(data_path):
    with open(data_path, "r") as f:
        data = json.load(f)
    return data


def sample_camera_rays(data, number_of_rays, num_samples, delta_step, even_spread, camera_ray):
    if even_spread:
        number_of_rays = int(np.round(np.sqrt(number_of_rays)) ** 2)

    # Preallocate memory for ray_directions and camera_positions
    num_frames = len(data["frames"])
    ray_directions = torch.zeros((num_frames * number_of_rays, 3)) if not camera_ray else torch.zeros((num_frames, 3))
    camera_positions = torch.zeros((num_frames, 3))

    for i in range(num_frames):
        transform_matrix, rotation, file_path, camera_angle_x = get_data_from_index(data, i)

        if camera_ray:
            current_ray_directions = transform_matrix[:3, 2].unsqueeze(0) * -1
            ray_directions[i] = current_ray_directions
        else:
            current_ray_directions = generate_rays(number_of_rays, transform_matrix, camera_angle_x,
                                                   even_spread=even_spread)
            ray_directions[i * number_of_rays:(i + 1) * number_of_rays] = current_ray_directions

        camera_positions[i] = transform_matrix[:3, 3]

    delta_forsamples = delta_step * torch.arange(num_samples + 1)[1:].repeat(number_of_rays * len(data["frames"])) \
        .unsqueeze(1)

    camera_positions_forsamples = torch.repeat_interleave(camera_positions, num_samples * number_of_rays, 0)

    ray_directions_forsamples = torch.repeat_interleave(ray_directions, num_samples, 0)
    samples_interval = camera_positions_forsamples + ray_directions_forsamples * delta_forsamples

    return samples_interval, camera_positions, ray_directions


def load_image_data(data_folder, object_folder):
    with open(f"{data_folder}/{object_folder}/transforms_train.json", "r") as f:
        data = json.load(f)
    imgs = [Image.open(f'{data_folder}/{object_folder}/train/{frame["file_path"].split("/")[-1]}.png') for frame in
            data["frames"]]
    imgs = np.array([np.array(img) for img in imgs])
    imgs = torch.tensor(imgs, dtype=torch.float)
    imgs = (imgs / 255)

    return data, imgs


def normalize_samples_for_indecies(grid_indices, samples_interval, points_distance):
    return (samples_interval - grid_indices.min(0)[0]) / points_distance


def trilinear_interpolation(normalized_samples_for_indecies, selected_points, grid_cells):
    """
    The input tensors:
        :param    normalized_samples_for_indecies has a shape of (N, 3), which represents N 3D points.
        :param    selected_points has a shape of (N, 8, 3), which represents 8 corner points for each of the N 3D points.
        :param    grid_cells has a shape of (X, Y, Z, 3), which represents a 3D grid with XxYxZ cells, where each cell has a 3D value associated with it.

    selected_points is of this form:
        [['ceil_x', 'ceil_y', 'ceil_z'],
         ['ceil_x', 'ceil_y', 'floor_z'],
         ['ceil_x', 'floor_y', 'ceil_z'],
         ['ceil_x', 'floor_y', 'floor_z'],
         ['floor_x', 'ceil_y', 'ceil_z'],
         ['floor_x', 'ceil_y', 'floor_z'],
         ['floor_x', 'floor_y', 'ceil_z'],
         ['floor_x', 'floor_y', 'floor_z']]

    1. selection0 and selection1:
        These two tensors are created by reshaping the last 4 and first 4 corner points of each cell in selected_points, respectively.
        Then, they extract the values from grid_cells corresponding to these corner points.

    2. interpolation_frac_step1:
        This tensor represents the interpolation fraction for the first dimension.
        The code multiplies selection1 by this fraction and selection0 by its complement (1 - fraction) and adds them together, yielding newstep.

    3. inteplation_frac_step2:
        This tensor represents the interpolation fraction for the second dimension.
        The code performs interpolation for the second dimension using the values computed in step 3 and updates newstep.

    4. inteplation_frac_step3:
        This tensor represents the interpolation fraction for the third dimension.
        The code performs interpolation for the third dimension using the values computed in step 4 and updates newstep.

    The final newstep tensor contains the interpolated values at the original input points based on the grid values.
    """

    inteplation_frac = torch.frac(normalized_samples_for_indecies)

    selection0 = grid_cells[selected_points[:, 4:, 0], selected_points[:, 4:, 1], selected_points[:, 4:, 2]]
    selection1 = grid_cells[selected_points[:, :4, 0], selected_points[:, :4, 1], selected_points[:, :4, 2]]

    inteplation_frac_step1 = inteplation_frac[:, 0].unsqueeze(-1).unsqueeze(-1)
    newstep = (selection1 * inteplation_frac_step1) + (selection0 * (1 - inteplation_frac_step1))

    inteplation_frac_step2 = inteplation_frac[:, 1].unsqueeze(-1).unsqueeze(-1)
    newstep = (newstep[:, :2] * inteplation_frac_step2) + (newstep[:, 2:] * (1 - inteplation_frac_step2))

    inteplation_frac_step3 = inteplation_frac[:, 2].unsqueeze(-1)
    newstep = (newstep[:, 0] * inteplation_frac_step3) + (newstep[:, 1] * (1 - inteplation_frac_step3))

    return newstep


def find_out_of_bound(indecies, grid):
    idx1, idx2, idx3 = indecies[:, 0], indecies[:, 1], indecies[:, 2]

    # find which points are outside the grid
    X, Y, Z, N = grid.shape
    idx1_outofbounds = (idx1 < X) & (idx1 >= 0)
    idx2_outofbounds = (idx2 < Y) & (idx2 >= 0)
    idx3_outofbounds = (idx3 < Z) & (idx3 >= 0)
    outofbounds = idx1_outofbounds & idx2_outofbounds & idx3_outofbounds

    return outofbounds


def fix_out_of_bounds(indecies, grid):
    idx1, idx2, idx3 = indecies[:, 0], indecies[:, 1], indecies[:, 2]
    X, Y, Z, N = grid.shape
    idx1 %= X
    idx2 %= Y
    idx3 %= Z

    return idx1, idx2, idx3


def get_nearest_voxels(normalized_samples_for_indecies, grid):
    indecies = torch.round(normalized_samples_for_indecies).to(torch.long)
    outofbounds = find_out_of_bound(indecies, grid)
    idx1, idx2, idx3 = fix_out_of_bounds(indecies, grid)
    return grid[idx1, idx2, idx3], outofbounds


def collect_cell_information_via_indices(normalized_samples_for_indecies, B):
    outofbounds = find_out_of_bound(normalized_samples_for_indecies, B)
    edge_matching_points = get_grid_points_indices(normalized_samples_for_indecies)
    A = edge_matching_points.reshape([edge_matching_points.shape[0] * edge_matching_points.shape[1],
                                      edge_matching_points.shape[2]])

    # Use advanced indexing to get the values
    idx1, idx2, idx3 = fix_out_of_bounds(A, B)
    output = B[idx1, idx2, idx3]
    return output, outofbounds



def fit(transform_matricies, camera_angle_x, imgs, number_of_rays, num_samples, delta_step, even_spread,
        camera_ray, points_distance, gridsize):
    samples_interval, pixels_to_rays, camera_positions, ray_directions = sample_camera_rays_batched(
        transform_matricies=transform_matricies,
        camera_angle_x=camera_angle_x,
        imgs=imgs,
        number_of_rays=number_of_rays,
        num_samples=num_samples,
        delta_step=delta_step,
        even_spread=even_spread,
        camera_ray=camera_ray
    )

    grid_indices, grid_cells, meshgrid, grid_grid = get_grid(gridsize[0], gridsize[1], gridsize[2],
                                                             points_distance=points_distance, info_size=4)
    normalized_samples_for_indecies = normalize_samples_for_indecies(grid_indices, samples_interval, points_distance)
    selected_points, mask = collect_cell_information_via_indices(normalized_samples_for_indecies, meshgrid)
    interpolated_points = trilinear_interpolation(normalized_samples_for_indecies, selected_points, grid_cells)

    return


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


def compute_alpha_weighted_pixels(samples):
    """

    :param samples: a tensor of size [Number of cameras, number of rays, number of samples along the rays]
    :return: a tensor of size [Number of cameras, number of rays,]
    compute the color of a pixel given the alphas of the grid
    """

    # compute sample weight using the alpha channels;
    alpha = samples[..., -1]
    padded_nearest_alpha = torch.cat([torch.zeros_like(alpha[..., :1]), alpha], dim=2)
    cumprod = (1 - padded_nearest_alpha[..., :-1]).cumprod(dim=2)
    weights = (alpha * cumprod).unsqueeze(-1)

    # multiply the weights by the color channels and sum, do the same to get the final alphas
    samples = (samples[..., :-1] * weights).sum(2)
    final_alpha = weights.sum(dim=2)

    # Combine the pixel colors and final alpha values
    res = torch.cat([samples, final_alpha], dim=2)
    return res


def inference_test_voxels():
    # parameters
    number_of_rays = 40000
    num_samples = 600
    delta_step = 0.01
    even_spread = True
    camera_ray = False
    points_distance = 0.5  # *2*10
    gridsize = [5, 5, 5]  # 64

    # loading data
    data_folder = r"D:\9.programming\Plenoxels\data"
    object_folders = ['chair', 'drums', 'ficus', 'hotdog', 'lego', 'materials', 'mic', 'ship']
    object_folder = object_folders[0]
    data, imgs = load_image_data(data_folder, object_folder)
    transform_matricies, file_paths, camera_angle_x = load_data(data)

    # generate grid
    grid_indices, grid_cells, meshgrid, grid_grid = get_grid(gridsize[0], gridsize[1], gridsize[2],
                                                             points_distance=points_distance, info_size=4)

    # draw voxels to create an image
    with torch.no_grad():
        grid_cells[2, 2, 0, 0]   = 1.0  # 0.498
        grid_cells[2, 2, 1, 0]   = 1.0
        grid_cells[3, 2, 0, 1:3] = 1.0
        grid_cells[2, 3, 0, 1]   = 1.0
        grid_cells[1, 2, 0, 2]   = 1.0
        grid_cells[2, 1, 0, :2]  = 1.0

        grid_cells[2, 2, 0, -1] = 1.0
        grid_cells[2, 2, 1, -1] = 0.5
        grid_cells[3, 2, 0, -1] = 0.2
        grid_cells[2, 3, 0, -1] = 0.7
        grid_cells[1, 2, 0, -1] = 1.0
        grid_cells[2, 1, 0, -1] = 0.5

    # choose an image to process
    img_index = 63
    imgs = imgs[img_index, :, :, :].unsqueeze(0)
    transform_matricies = transform_matricies[img_index, :, :].unsqueeze(0)

    # generate samples
    samples_interval, pixels_to_rays, camera_positions, ray_directions = sample_camera_rays_batched(
        transform_matricies=transform_matricies,
        camera_angle_x=camera_angle_x,
        imgs=imgs,
        number_of_rays=number_of_rays,
        num_samples=num_samples,
        delta_step=delta_step,
        even_spread=even_spread,
        camera_ray=camera_ray
    )

    # compute closest grid points to samples
    normalized_samples_for_indecies = normalize_samples_for_indecies(grid_indices, samples_interval, points_distance)

    # -- knn via Manhattan distance --
    nearest, mask = get_nearest_voxels(normalized_samples_for_indecies, grid_cells)
    nearest = nearest * mask.unsqueeze(-1)  # zero the samples landing outside the grid

    # -- interpolate 8 closest points --
    # selected_points, mask = collect_cell_information_via_indices(normalized_samples_for_indecies, meshgrid)
    # selected_points = selected_points.reshape([int(selected_points.shape[0] / 8), 8, 3])
    # nearest = trilinear_interpolation(normalized_samples_for_indecies, selected_points, grid_cells)
    # nearest = nearest * mask.unsqueeze(-1)

    # find pixels color
    nearest = nearest.reshape(1, number_of_rays, num_samples, 4)
    pixels_color = compute_alpha_weighted_pixels(nearest)

    # -- reshape and normalize an image and show --
    pixels_color = pixels_color.detach().numpy()
    pixels_color = (pixels_color * 255).round().clip(0, 255).astype(np.uint8)
    image = pixels_color.reshape(200, 200, 4)
    image = np.transpose(image, (1, 0, 2))

    # -- sanity test for image --
    # image = pixels_to_rays.reshape(200, 200, 4).numpy()
    # image = (image * 255).astype(np.uint8)
    # image = np.transpose(image, (1, 0, 2))

    plt.imshow(image)
    plt.show()

    return


def main():
    """
    This function reads a dataset of 3D object transformations, generates camera rays, and visualizes
    the rays in 3D space. The dataset contains information about frames, and the function processes
    each frame to extract relevant data. Rays are generated based on user-defined parameters, and
    the visualization is created using the generated rays and camera positions.

    Parameters:
    - data_folder (str): The folder containing the data files.
    - object_folders (List[str]): A list of folder names containing object-specific data.
    - number_of_rays (int): The desired number of rays to generate.
    - delta_step (float): The step size between samples along each ray.
    - num_samples (int): The number of samples to take along each ray.
    - even_spread (bool): If True, generates rays with an even angular spread; otherwise, uses a
                          random distribution.
    - camera_ray (bool): If True, only a single ray is generated for each frame, directly opposite
                         to the camera direction.

    Returns:
    None
    """
    number_of_rays = 16
    num_samples = 9#200
    delta_step = 0.5
    even_spread = True
    camera_ray = False
    points_distance = 0.5#*2*10
    gridsize = [16, 16, 16]#64

    # number_of_rays = 9
    # num_samples = 60  # 200
    # delta_step = .1
    # even_spread = True
    # camera_ray = False
    # points_distance = 0.5  # *2*10
    # gridsize = [5, 5, 5]  # 64

    data_folder = r"D:\9.programming\Plenoxels\data"
    object_folders = ['chair', 'drums', 'ficus', 'hotdog', 'lego', 'materials', 'mic', 'ship']
    object_folder = object_folders[0]
    data, imgs = load_image_data(data_folder, object_folder)

    grid_indices, grid_cells, meshgrid, grid_grid = get_grid(gridsize[0], gridsize[1], gridsize[2],
                                                             points_distance=points_distance, info_size=4)
    t0 = time.time()
    transform_matricies, file_paths, camera_angle_x = load_data(data)
    samples_interval, pixels_to_rays, camera_positions, ray_directions = sample_camera_rays_batched(
        transform_matricies=transform_matricies,
        camera_angle_x=camera_angle_x,
        imgs=imgs,
        number_of_rays=number_of_rays,
        num_samples=num_samples,
        delta_step=delta_step,
        even_spread=even_spread,
        camera_ray=camera_ray
    )
    normalized_samples_for_indecies = normalize_samples_for_indecies(grid_indices, samples_interval, points_distance)

    print(f"{samples_interval.shape=}\n{samples_interval[0]=}")
    print(f"{normalized_samples_for_indecies.shape=}\n{normalized_samples_for_indecies[0]=}")
    print(f"{normalized_samples_for_indecies.requires_grad=}")

    print(time.time() - t0)

    # -- visulize grid
    # visualize_rays_3d(ray_directions, [], grid_indices)

    # -- visulize camera rays and samples along rays
    # ray_positions = torch.repeat_interleave(camera_positions, number_of_rays, 0)
    # sampled_rays = samples_interval[num_samples*number_of_rays*10:num_samples*(number_of_rays*10 + number_of_rays)]
    # visualize_rays_3d(ray_directions, ray_positions, sampled_rays, grid_indices)
    # visualize_rays_3d(ray_directions, ray_positions, sampled_rays)

    # -- visulize grid around sampled points of ray
    index = 5
    # selected_points, mask = collect_cell_information_via_indices(normalized_samples_for_indecies, meshgrid)
    # selected_points = selected_points.reshape([int(selected_points.shape[0] / 8), 8, 3])
    # paper_visulization(index, grid_grid, num_samples, number_of_rays, selected_points, samples_interval,
    #                    ray_directions)
    # result = trilinear_interpolation(normalized_samples_for_indecies, selected_points, grid_cells)

    # -- visulize points on grid closest to sampled points of ray
    selected_points_voxels, outofbound_mask = get_nearest_voxels(normalized_samples_for_indecies, meshgrid)
    voxel_visulization(index, grid_grid, num_samples, number_of_rays, selected_points_voxels, samples_interval,
                       ray_directions)

    # print(f"{outofbound_mask.shape=}\n{outofbound_mask.unique(return_counts =True)}")


if __name__ == "__main__":
    printi("start")
    # main()
    inference_test_voxels()
    printi("end")
