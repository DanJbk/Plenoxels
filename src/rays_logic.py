import json
import torch
import numpy as np
import logging
import matplotlib.pyplot as plt
from logging import info as printi
from mpl_toolkits.mplot3d import Axes3D
from torch import Tensor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def get_points(grid_indices, samples_interval, points_distance):
    # (grid_indices - grid_indices.min(0)[0])/points_distance

    normalized_samples_for_indecies = ((samples_interval - grid_indices.min(0)[0]) / points_distance)

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


def collect_cell_information_via_indices(A, B):
    # A = torch.tensor([[0, 1, 8]]*K)  # Shape: [K, 3]
    # B = torch.randn(X, Y, Z, N)  # Shape: [X, Y, Z, N]

    X, Y, Z, N = B.shape

    # Define the desired column order
    column_order = torch.tensor([1, 0, 2])

    # Reorder the columns of A
    # A = torch.index_select(A, 1, column_order)

    # Reshape B to [X*Y*Z, N]
    B_flat = B.reshape(-1, N)  # Shape: [X*Y*Z, N]

    # Calculate flat indices from A
    indices = A[:, 0] * Y * Z + A[:, 1] * Z + A[:, 2]

    # Gather the elements from B_flat using the indices
    return torch.gather(B_flat, 0, indices.view(-1, 1).expand(-1, N))


def get_grid(sx, sy, sz, points_distance=0.5, info_size=3):
    grindx_indices, grindy_indices, grindz_indices = torch.arange(sx), torch.arange(sy), torch.arange(sz)

    coordsz, coordsx, coordsy = torch.meshgrid(grindz_indices, grindx_indices, grindy_indices, indexing='ij')

    meshgrid = torch.stack([coordsx, coordsy, coordsz]).T

    # center grid
    coordsx, coordsy, coordsz = coordsx - np.ceil(sx / 2) + 1, coordsy - np.ceil(sy / 2) + 1, coordsz - np.ceil(
        sz / 2) + 1

    # edit grid spacing
    coordsx, coordsy, coordsz = coordsx * points_distance, coordsy * points_distance, coordsz * points_distance

    # make it so no points of the grid are underground
    coordsz = coordsz - coordsz.min()

    grid_grid = torch.stack([coordsx, coordsy, coordsz]).T
    grid_coords = grid_grid.reshape(sx * sy * sz, 3)

    grid_cells = torch.zeros_like(grid_grid)

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
        for i in range(ray_directions.shape[0]):
            ax.quiver(origin_x[i], origin_y[i], origin_z[i],
                      ray_directions[i, 0], ray_directions[i, 1], ray_directions[i, 2],
                      color='b', alpha=0.5)

    for item, color in zip([red, green, orange], ['r', 'g', 'orange']):
        if not item is None:
            scatter_x = item[:, 0]
            scatter_y = item[:, 1]
            scatter_z = item[:, 2]
            ax.scatter(scatter_x, scatter_y, scatter_z, c=color, marker='o')

    # todo base limits on grid and/or samples
    ax.set_xlim3d(-5, 5)
    ax.set_ylim3d(-5, 5)
    ax.set_zlim3d(0, 5 * 2)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title("3D Visualization of Evenly Spread Ray Directions within Camera's FOV")
    plt.show()


def read_data(data_path):
    with open(data_path, "r") as f:
        data = json.load(f)
    return data


def sample_camera_rays(data, number_of_rays, num_samples, delta_step, even_spread, camera_ray):

    if even_spread:
        number_of_rays = int(np.round(np.sqrt(number_of_rays))**2)

    ray_directions = torch.zeros([])
    camera_positions = torch.zeros([])
    for i in range(len(data["frames"])):
        transform_matrix, rotation, file_path, camera_angle_x = get_data_from_index(data, i)

        if camera_ray:
            current_ray_directions = transform_matrix[:3, 2].unsqueeze(0) * -1

        else:
            current_ray_directions = generate_rays(number_of_rays, transform_matrix, camera_angle_x, even_spread=even_spread)

        ray_directions = current_ray_directions if len(ray_directions.shape) == 0 else torch.cat(
            [ray_directions, current_ray_directions], 0)
        camera_positions = transform_matrix[:3, 3].unsqueeze(0) if len(camera_positions.shape) == 0 else torch.cat(
            [camera_positions, transform_matrix[:3, 3].unsqueeze(0)], 0)

    delta_forsamples = delta_step*torch.arange(num_samples + 1)[1:].repeat(number_of_rays * len(data["frames"]))\
        .unsqueeze(1)

    camera_positions_forsamples = torch.repeat_interleave(camera_positions, num_samples*number_of_rays, 0)

    ray_directions_forsamples = torch.repeat_interleave(ray_directions, num_samples, 0)
    samples_interval = camera_positions_forsamples + ray_directions_forsamples * delta_forsamples

    return samples_interval, camera_positions, ray_directions


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
    number_of_rays = 9
    num_samples = 13
    delta_step = 0.3
    even_spread = True
    camera_ray = False
    points_distance = 0.9

    data_folder = "D:\9.programming\Plenoxels\data"
    object_folders = ['chair', 'drums', 'ficus', 'hotdog', 'lego', 'materials', 'mic', 'ship']

    with open(f"{data_folder}/{object_folders[0]}/transforms_train.json", "r") as f:
        data = json.load(f)

    samples_interval, camera_positions, ray_directions = sample_camera_rays(data,
                                                                            number_of_rays=number_of_rays,
                                                                            num_samples=num_samples,
                                                                            delta_step=delta_step,
                                                                            even_spread=even_spread,
                                                                            camera_ray=camera_ray
                                                                            )

    grid_indices, grid_cells, meshgrid, grid_grid = get_grid(19, 19, 19, points_distance=points_distance, info_size=4)

    # visualize_rays_3d(ray_directions, camera_positions, grid_indices,samples_interval)
    # visualize_rays_3d(ray_directions, torch.repeat_interleave(camera_positions, number_of_rays, 0),
    #                   samples_interval[num_samples*number_of_rays*10:num_samples*(number_of_rays*10 + 2)])

    edge_matching_points = get_points(grid_indices, samples_interval, points_distance)
    edge_matching_points = edge_matching_points.reshape([edge_matching_points.shape[0] * edge_matching_points.shape[1],
                                                         edge_matching_points.shape[2]])
    selected_points = collect_cell_information_via_indices(edge_matching_points, meshgrid)
    selected_points = selected_points.view([int(selected_points.shape[0] / 8), 8, 3])

    # todo put in function and fix cases where point is out of grid (y,x,z)
    index = 700
    temp = torch.cat(
        [grid_grid[tuple(s.tolist())].unsqueeze(0) for i in [index * num_samples + i for i in range(num_samples)] for s
         in selected_points[i]])
    temp2 = torch.cat([samples_interval[i].unsqueeze(0) for i in
                       [index * num_samples + i for i in range(num_samples)]], 0)
    visualize_rays_3d(ray_directions, [], temp, temp2)


if __name__ == "__main__":
    printi("start")
    main()
    printi("end")
