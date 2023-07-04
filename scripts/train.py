import argparse

import torch
from torch.optim import SGD
from torch.optim import Adam
from torch.optim import RMSprop
from torch.nn.functional import mse_loss
from tqdm import tqdm

from src.data_processing import load_data, load_image_data_from_path
from src.grid_functions import get_nearest_voxels, generate_grid, average_pool3d_grid
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


def tv_loss(input_tensor):
    """
    Computes L2 Total Variation (TV) loss for a 3D tensor of size [x,y,z,4].
    Args:
        input_tensor (torch.Tensor): 3D tensor of shape [x, y, z, 4].
    Returns:
        tv_loss (torch.Tensor): Scalar tensor containing L2 TV loss value.
    """
    # Compute the spatial gradients of the tensor along x, y, and z axes

    gradient_x = (input_tensor[:, :-1, :, :] - input_tensor[:, 1:, :, :])
    gradient_y = (input_tensor[:, :, :-1, :] - input_tensor[:, :, 1:, :])
    gradient_z = (input_tensor[:-1, :, :, :] - input_tensor[1:, :, :, :])

    gradient_x = gradient_x.pow(2)
    gradient_y = gradient_y.pow(2)
    gradient_z = gradient_z.pow(2)

    # L2 norms using the given weighting factors
    tv_loss = torch.sqrt(gradient_x.sum() + gradient_y.sum() + gradient_z.sum())

    return tv_loss


def fit(gridsize, points_distance_original, number_of_rays, num_samples, delta_step, lr, tv, beta, steps, even_spread, path,
        transform_path, save_path, device):

    camera_ray = False

    data, imgs = load_image_data_from_path(path, transform_path)
    transform_matricies, file_paths, camera_angle_x = load_data(data)
    transform_matricies, imgs = transform_matricies.to(device), imgs.to(device)

    # transform_matricies, imgs = transform_matricies[[10, 20, 30, 40, 50, 60, 70, 80], ...], imgs[[10, 20, 30, 40, 50, 60, 70, 80], ...]
    transform_matricies, imgs = transform_matricies[[10, 68, 41], ...], imgs[[10, 68, 41], ...]

    grid_indices, grid_cells_full, meshgrid, grid_grid_original = generate_grid(gridsize[0], gridsize[1], gridsize[2],
                                                                  points_distance=points_distance_original, info_size=4,
                                                                  device=device)


    with torch.no_grad():
        # original_grid_colors = torch.rand_like(grid_cells_full[:, :, :, :-1])
        original_grid_colors = torch.zeros_like(grid_cells_full[:, :, :, :-1])
        grid_cells_full[:, :, :, :-1] = original_grid_colors

    optimizer = Adam([grid_cells_full], lr=lr)



    loss_hist = []
    beta_loss_hist = []
    tv_loss_hist = []

    pbar = tqdm(total=steps, desc="Processing items")

    # receptive_fields = [i for i in range(1,51) if i%2 != 0 ]
    # start_receptive_fields = [i for i in range(len(receptive_fields))]

    receptive_fields = [i for i in range(3, 51, 2) if i % 2 != 0]
    start_receptive_fields = [i * 15 for i in range(1 + len(receptive_fields), 0, -1)]

    for i in range(steps):

        # ---

        receptive_field_size = max(a if i < b else 0 for a, b in zip(receptive_fields, start_receptive_fields))
        if i % 10 == 0:
            print(f"{receptive_field_size=}")
        if receptive_field_size > 1:

            stride = max(1, receptive_field_size // 4)

            start_index = int(receptive_field_size / 2)

            grid_grid = grid_grid_original[start_index::stride, start_index::stride,
                        start_index::stride]
            grid_indices = grid_grid.reshape(-1, 3)

            points_distance = points_distance_original * stride #(receptive_field_size - 1)

            grid_cells = average_pool3d_grid(grid_cells_full, receptive_field_size=receptive_field_size, stride=stride)
        else:
            grid_cells = grid_cells_full
            grid_indices = grid_grid_original.reshape(-1, 3)
            points_distance = points_distance_original

        # ---

        # generate samples
        samples_interval, pixels_to_rays, camera_positions, ray_directions = sample_camera_rays_batched(
            transform_matrices=transform_matricies,
            camera_angle_x=camera_angle_x,
            imgs=imgs,
            number_of_rays=number_of_rays,
            num_samples=num_samples,
            delta_step=delta_step,
            even_spread=even_spread,
            camera_ray=camera_ray,
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

        # compute loss
        mseloss = mse_loss(pixels_color, pixels_to_rays)
        loss = mseloss

        loss_detached = mseloss.cpu().detach()
        update_string.append(f"closs:{loss_detached:10.6f}")
        loss_hist.append(loss_detached)

        if tv > 0 and receptive_field_size == 0:
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

        # if True:
        #     # gridmseloss = 50*grid_cells[:, :, :, -1].flatten().clamp(0, 1).mean()
        #     nshape = nearest.shape
        #     linspace = torch.linspace(0, 1, nshape[2], device=device).unsqueeze(0).unsqueeze(0)
        #     linespace_tens = (1 - linspace.expand(nshape[0], nshape[1], nshape[2]))**2
        #     gridmseloss = 0.5*(nearest[:, :, :, -1].clip(0, 1)*linespace_tens).sum(-1).mean()
        #     loss += gridmseloss
        #     gridmseloss_detached = gridmseloss.detach().cpu()
        #     update_string.append(f"gridmseloss:{gridmseloss_detached:10.6f}")
        #     tv_loss_hist.append(gridmseloss_detached)

        # optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update bar
        pbar.update(1)
        pbar.set_description(" ".join(update_string))

    # save grid and parameters
    torch.save({
        "grid": grid_cells_full.detach().cpu(),
        "param": {
            "device": device,
            "number_of_rays": number_of_rays,
            "num_samples": num_samples,
            "delta_step": delta_step,
            "even_spread": even_spread,
            "camera_ray": camera_ray,
            "points_distance": points_distance_original,
            "gridsize": gridsize,
        },
    }, save_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='visualize camera positions around a grid.')
    parser.add_argument('--device', default="auto", help='device to run on (cpu, cuda, auto)')
    parser.add_argument('--save_path', default="src/grid_cells_trained.pth", help='path to trained grid')
    parser.add_argument('--ray_num', default=128, type=int, help='amount of rays from each camera during training')
    parser.add_argument('--sample_num', default=600, type=int, help='number of samples along a ray')
    parser.add_argument('--path', default="data/mic/train", help='path to dataset')
    parser.add_argument('--transform_path', default="data/mic/transforms_train.json", help='path to json file')
    parser.add_argument('--gridsize', default=-1, type=int, help='specify cubic size of grid as a single number')
    parser.add_argument('--grid_dim', default=[256, 256, 256], nargs='+', help='size of grid, write as "x y z"')
    parser.add_argument('--points_distance', default=0.0125, type=float, help='delta distance between points in grid')
    parser.add_argument('--delta_step', default=0.0125, type=float, help='distance between samples along a ray')
    parser.add_argument('--lr', default=0.0075, type=float, help='learning rate')
    parser.add_argument('--tv', default=1e-5, type=float, help='tv loss strength (set to 0 to disable)')
    parser.add_argument('--beta', default=5e-3, type=float, help='tv loss strength (set to 0 to disable)')
    parser.add_argument('--steps', default=500, type=int, help='training steps')
    parser.add_argument('--even_spread', default=False, type=bool,  help='whether to set spread rays evenly. '
                                                                         'the ray number needs to have a natural '
                                                                         'square root')
    args = parser.parse_args()

    main(args)
