
from src.data_processing import load_data, load_image_data_from_path
from src.ray_sampling import sample_camera_rays_batched
from src.visualization import visualize_rays_3d
from src.grid_functions import generate_grid

import argparse
import torch


def main(args):

    device = args.device
    if device == "auto":
        device = "cpu" if torch.cuda.is_available() else "cuda:0"

    path = args.path
    transform_path = args.transform_path
    points_distance = args.point_distance

    gridsize = [args.gridsize] * 3
    if -1 in gridsize:
        gridsize = [int(n) for n in args.grid_dim]

    view_grid_cameras(path, transform_path, gridsize, points_distance, device)


def view_grid_cameras(path, transform_path, gridsize, points_distance, device):

    number_of_rays = 9

    data, imgs = load_image_data_from_path(path, transform_path)
    transform_matrices, file_paths, camera_angle_x = load_data(data)
    transform_matrices, imgs = transform_matrices.to(device), imgs.to(device)

    samples_interval, pixels_to_rays, camera_positions, ray_directions = sample_camera_rays_batched(
        transform_matrices=transform_matrices,
        camera_angle_x=camera_angle_x,
        imgs=imgs,
        number_of_rays=number_of_rays,
        num_samples=0,
        delta_step=0,
        even_spread=True,
        camera_ray=False,
        device=device
    )
    grid_indices, grid_cells, meshgrid, grid_grid = generate_grid(gridsize[0], gridsize[1], gridsize[2],
                                                                  points_distance=points_distance, info_size=4,
                                                                  device=device)
    ray_positions = torch.repeat_interleave(camera_positions, number_of_rays, 0)

    if any(map(lambda x: x >= 8, gridsize)):
        downsample_factor = int(grid_grid.shape[0] / 8)
        grid_grid = grid_grid[::downsample_factor, ::downsample_factor, ::downsample_factor, :]
        grid_grid = grid_grid.reshape(grid_grid.shape[0] * grid_grid.shape[1] * grid_grid.shape[2], 3).to("cpu")

    visualize_rays_3d(ray_directions.cpu(), ray_positions.cpu(), grid_grid.cpu())


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='visualize camera positions around a grid.')
    parser.add_argument('--device', default="auto", help='device to run on (cpu, cuda, auto)')
    parser.add_argument('--gridsize', default=-1, type=int, help='specify cubic size of grid as a single number')
    parser.add_argument('--grid_dim', default=[64, 64, 64], nargs='+', help='size of grid, "x y z')
    parser.add_argument('--point_distance', default=0.05, type=float, help='distance from point to point')
    parser.add_argument('--path', default="data/chair/train", help='path to dataset')
    parser.add_argument('--transform_path', default="data/chair/transforms_train.json", help='path to dataset')
    args = parser.parse_args()

    main(args)
