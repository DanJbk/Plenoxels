import argparse

import torch
from torch.optim import SGD
from torch.optim import Adam
from torch.optim import RMSprop
from torch.nn.functional import mse_loss
from tqdm import tqdm

from src.data_processing import load_data, load_image_data_from_path
from src.grid_functions import get_nearest_voxels, generate_grid
from src.ray_sampling import compute_alpha_weighted_pixels, normalize_samples_for_indecies, sample_camera_rays_batched


def main(args):

    device = args.device
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    gridsize = [args.gridsize] * 3
    if -1 in gridsize:
        gridsize = [int(n) for n in args.grid_dim]

    points_distance = args.points_distance
    even_spread = args.even_spread
    number_of_rays = args.ray_num
    num_samples = args.sample_num  # 200
    delta_step = args.delta_step  # 200
    transform_path = args.transform_path
    path = args.path
    save_path = args.save_path
    lr = args.lr
    tv = args.tv
    beta = args.beta

    steps = args.steps

    fit(gridsize, points_distance, number_of_rays, num_samples, delta_step, lr, tv, beta, steps, even_spread, path,
        transform_path, save_path, device)


def tv_loss(voxel_grid):

    # Compute gradient of voxel grid
    grad_x = (voxel_grid[:, :, :-1, :] - voxel_grid[:, :, 1:, :])
    grad_y = (voxel_grid[:, :-1, :, :] - voxel_grid[:, 1:, :, :])
    grad_z = (voxel_grid[:-1, :, :, :] - voxel_grid[1:, :, :, :])

    # Compute TV loss
    tv_loss = torch.mean(grad_x * grad_x) + torch.mean(grad_y * grad_y) + torch.mean(grad_z * grad_z)

    return tv_loss


def fit(gridsize, points_distance, number_of_rays, num_samples, delta_step, lr, tv, beta, steps, even_spread, path,
        transform_path, save_path, device):

    camera_ray = False

    data, imgs = load_image_data_from_path(path, transform_path)
    transform_matricies, file_paths, camera_angle_x = load_data(data)
    transform_matricies, imgs = transform_matricies.to(device), imgs.to(device)

    # transform_matricies, imgs = transform_matricies[[10, 20, 30, 40, 50, 60, 70, 80], ...], imgs[[10, 20, 30, 40, 50, 60, 70, 80], ...]

    grid_indices, grid_cells, meshgrid, grid_grid = generate_grid(gridsize[0], gridsize[1], gridsize[2],
                                                                  points_distance=points_distance, info_size=4,
                                                                  device=device)
    with torch.no_grad():
        original_grid_colors = torch.rand_like(grid_cells[:, :, :, :-1])
        grid_cells[:, :, :, :-1] = original_grid_colors

    optimizer = Adam([grid_cells], lr=lr)

    loss_hist = []
    beta_loss_hist = []
    tv_loss_hist = []

    pbar = tqdm(total=steps, desc="Processing items")
    for i in range(steps):

        # generate samples
        samples_interval, pixels_to_rays, camera_positions, ray_directions = sample_camera_rays_batched(
            transform_matrices=transform_matricies,
            camera_angle_x=camera_angle_x,
            imgs=imgs,
            number_of_rays=number_of_rays,
            num_samples=num_samples,
            delta_step=delta_step,
            even_spread=even_spread,
            camera_ray=False,
            device=device
        )

        normalized_samples_for_indecies = normalize_samples_for_indecies(grid_indices, samples_interval,
                                                                         points_distance)

        # assign voxel to sample
        nearest, mask = get_nearest_voxels(normalized_samples_for_indecies, grid_cells.clamp(0, 1))
        nearest = nearest * mask.unsqueeze(-1)  # zero the samples that are outside the grid

        # find pixels color
        nearest = nearest.reshape(imgs.shape[0], number_of_rays, num_samples, 4)
        pixels_color = compute_alpha_weighted_pixels(nearest)
        color_shape = pixels_color.shape
        pixels_color = pixels_color.reshape([color_shape[0] * color_shape[1], color_shape[2]])

        update_string = []

        mseloss = mse_loss(pixels_color, pixels_to_rays)
        loss = mseloss

        loss_detached = mseloss.cpu().detach()
        update_string.append(f"closs:{loss_detached:10.6f}")
        loss_hist.append(loss_detached)

        if tv > 0:
            tvloss = tv*tv_loss(grid_cells)
            loss += tvloss
            tvloss_detached = tvloss.detach().cpu()
            update_string.append(f"tvloss:{tvloss_detached:10.6f}")
            tv_loss_hist.append(tvloss_detached)

        if beta > 0:
            epsilon = 0.0001
            betaloss = beta * (torch.log(nearest[:, :, :, -1] + epsilon) + torch.log(1 - nearest[:, :, :, -1] +
                                                                                     epsilon)).mean()
            loss += betaloss
            betaloss_detached = betaloss.detach().cpu()
            update_string.append(f"betaloss:{betaloss_detached:10.6f}")
            beta_loss_hist.append(betaloss_detached)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pbar.update(1)
        pbar.set_description(" ".join(update_string))

    # save grid and parameters
    torch.save({
        "grid": grid_cells.detach().cpu(),
        "param": {
            "device": device,
            "number_of_rays": number_of_rays,
            "num_samples": num_samples,
            "delta_step": delta_step,
            "even_spread": even_spread,
            "camera_ray": camera_ray,
            "points_distance": points_distance,
            "gridsize": gridsize,
        },
    }, save_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='visualize camera positions around a grid.')
    parser.add_argument('--device', default="auto", help='device to run on (cpu, cuda, auto)')
    parser.add_argument('--save_path', default="src/grid_cells_trained.pth", help='path to data')
    parser.add_argument('--ray_num', default=128, type=int, help='resolution of image, should (width/height) squared')
    parser.add_argument('--sample_num', default=600, type=int, help='number of samples along a ray')
    parser.add_argument('--path', default="data/ship/train", help='path to dataset')
    parser.add_argument('--transform_path', default="data/ship/transforms_train.json", help='path to json file')
    parser.add_argument('--gridsize', default=-1, type=int, help='specify cubic size of grid as a single number')
    parser.add_argument('--grid_dim', default=[256, 256, 256], nargs='+', help='size of grid, "x y z"')
    parser.add_argument('--points_distance', default=0.0125, type=float, help='delta distance between points in grid')
    parser.add_argument('--delta_step', default=0.0125, type=float, help='distance between samples along a ray')
    parser.add_argument('--lr', default=0.0075, type=float, help='learning rate')
    parser.add_argument('--tv', default=2.5, type=float, help='tv loss strength (set to 0 to disable)')
    parser.add_argument('--beta', default=2.5, type=float, help='tv loss strength (set to 0 to disable)')
    parser.add_argument('--steps', default=500, type=int, help='training steps')
    parser.add_argument('--even_spread', default=False, type=bool,  help='set to false for best results')
    args = parser.parse_args()

    main(args)
