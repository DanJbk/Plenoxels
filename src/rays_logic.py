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

from tqdm.auto import tqdm
from torch.optim import SGD
from torch.optim import Adam
from torch.optim import RMSprop
from torch.nn.functional import mse_loss
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.lr_scheduler import LinearLR

from src.data_processing import load_image_data, load_data
from src.grid_functions import get_grid, get_nearest_voxels, collect_cell_information_via_indices
from src.ray_sampling import sample_camera_rays_batched, normalize_samples_for_indecies
from src.visualization import paper_visulization, visualize_rays_3d, voxel_visulization

log_color = "\x1b[32;1m"
logging.basicConfig(level=logging.INFO, format=f'{log_color}%(asctime)s - %(message)s')

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


def tv_loss(voxel_grid):
    # Compute gradient of voxel grid
    grad_x = (voxel_grid[:, :, :-1, -1] - voxel_grid[:, :, 1:, -1])
    grad_y = (voxel_grid[:, :-1, :, -1] - voxel_grid[:, 1:, :, -1])
    grad_z = (voxel_grid[:-1, :, :, -1] - voxel_grid[1:, :, :, -1])

    # Compute TV loss
    tv_loss = torch.mean(grad_x ** 2) + torch.mean(grad_y ** 2) + torch.mean(grad_z ** 2)

    return tv_loss


def fit():
    device = "cuda"
    number_of_rays = 128
    num_samples = 600  # 200
    delta_step = 0.0125
    even_spread = False
    camera_ray = False
    points_distance = 0.0125  # *2*10
    gridsize = [256, 256, 256]

    lr = 0.0075

    # loading data and sending to device
    data_folder = r"D:\9.programming\Plenoxels\data"
    object_folders = ['chair', 'drums', 'ficus', 'hotdog', 'lego', 'materials', 'mic', 'ship']
    object_folder = object_folders[0]
    data, imgs = load_image_data(data_folder, object_folder, split="train")
    transform_matricies, file_paths, camera_angle_x = load_data(data)
    transform_matricies, imgs = transform_matricies.to(device), imgs.to(device)

    # transform_matricies, imgs = transform_matricies[[10, 20, 30, 40, 50, 60, 70, 80], ...], imgs[[10, 20, 30, 40, 50, 60, 70, 80], ...]

    grid_indices, grid_cells, meshgrid, grid_grid = get_grid(gridsize[0], gridsize[1], gridsize[2],
                                                             points_distance=points_distance, info_size=4,
                                                             device=device)
    with torch.no_grad():
        original_grid_colors = torch.rand_like(grid_cells[:, :, :, :-1])
        grid_cells[:, :, :, :-1] = original_grid_colors
        # grid_cells[:, :, :, :-1] = 1.0

    optimizer = Adam([grid_cells], lr=lr)

    # lr_decay = lambda step: lr if step < 200 else lr / 4
    # lr_decay = lambda step: max(0.01, lr * 0.5 ** (step / 2500))
    # scheduler = LambdaLR(optimizer, lr_lambda=lr_decay)
    # scheduler = LinearLR(optimizer, start_factor=0.9, total_iters=400, verbose=True)

    steps = 500
    loss_hist = []
    sparsity_loss_hist = []
    tv_loss_hist = []



    pbar = tqdm(total=steps, desc="Processing items")
    for i in range(steps):
        # generate samples
        samples_interval, pixels_to_rays, camera_positions, ray_directions = sample_camera_rays_batched(
            transform_matricies=transform_matricies,
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

        # -- knn via Manhattan distance --
        nearest, mask = get_nearest_voxels(normalized_samples_for_indecies, grid_cells.clamp(0, 1))
        nearest = nearest * mask.unsqueeze(-1)  # zero the samples landing outside the grid

        # find pixels color
        nearest = nearest.reshape(imgs.shape[0], number_of_rays, num_samples, 4)
        pixels_color = compute_alpha_weighted_pixels(nearest)
        color_shape = pixels_color.shape

        pixels_color = pixels_color.reshape([color_shape[0] * color_shape[1], color_shape[2]])

        epsilon = 0.0001
        # epsilon = 0.0001
        # sparsity_loss = 0.005 * (torch.log(nearest[:, :, :, -1] + epsilon) + torch.log(
        #     1 - nearest[:, :, :, -1] + epsilon)).mean()  # todo check if needed
        # tvloss = 2.5*tv_loss(grid_cells)

        tvloss = torch.tensor([0])
        sparsity_loss = torch.tensor([0])
        # sparsity_loss = 0.0005*(torch.log(grid_cells[..., -1].clamp(epsilon, 1) ) + torch.log(1 - grid_cells[..., -1].clamp(0, 1-epsilon))).mean() # todo check if needed

        # l2_loss = 0#10*(grid_cells[:,:,:,-1]**2).mean()
        mseloss = mse_loss(pixels_color, pixels_to_rays)
        loss = mseloss #+ tvloss #+ sparsity_loss  #

        optimizer.zero_grad()
        loss.backward()

        # grid_cells.grad[grid_cells.grad.norm(dim=3,p=2) < 0.00001] = 0

        optimizer.step()

        # if i > 300:
        #     scheduler.step()

        # with torch.no_grad():
        #     grid_cells.clamp_(min=0, max=1)

        # print(grid_cells.grad.min())

        loss_detached = mseloss.cpu().detach()
        sparsity_loss_detached = sparsity_loss.detach().cpu()
        tvloss_detached = tvloss.detach().cpu()
        loss_hist.append(loss_detached)
        sparsity_loss_hist.append(sparsity_loss_detached)
        tv_loss_hist.append(tvloss_detached)

        pbar.update(1)

        pbar.set_description(f"loss:{loss_detached} sparcity: {sparsity_loss_detached} tv: {tvloss_detached}")

    # with torch.no_grad():
    #     pixel_color_differences = torch.abs(original_grid_colors - grid_cells[:, :, :, :-1]).mean(3).detach().cpu()

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
            # "color_difference": pixel_color_differences,
        },
    }, "grid_cells_trained.pth")

    """
    import numpy as np
    from sklearn.linear_model import LinearRegression

    # Generate some sample data
    x = np.arange(len(loss_hist[2000:]))
    y = np.array(loss_hist[2000:])

    # Take the logarithm of y
    ylog = np.log(y)

    # Fit a linear regression model
    model = LinearRegression().fit(x.reshape(-1, 1), ylog)

    # Extract the slope and intercept of the line
    m = -model.coef_[0]
    c = model.intercept_

    # Print the equation of the line
    print(f"y = {np.exp(c):.4f} * exp(-{m} * x)")
    """


    return


def inference_test_voxels(grid_cells_path="", transparency_threshold=0.2, imgindex=26, do_threshold=False):
    # parameters
    # device = "cuda"
    # number_of_rays = 2500
    # num_samples = 600
    # delta_step = 0.01
    # even_spread = False
    # camera_ray = False
    # points_distance = 0.5  # *2*10
    # gridsize = [5, 5, 5]  # 64

    device = "cuda"
    number_of_rays = 40000
    num_samples = 600  # 200
    delta_step = 0.0120
    even_spread = True
    camera_ray = False
    points_distance = 0.0125  # *2*10
    gridsize = [256, 256, 256]

    # loading data
    data_folder = r"D:\9.programming\Plenoxels\data"
    object_folders = ['chair', 'drums', 'ficus', 'hotdog', 'lego', 'materials', 'mic', 'ship']
    object_folder = object_folders[0]
    data, imgs = load_image_data(data_folder, object_folder, split="test")
    transform_matricies, file_paths, camera_angle_x = load_data(data)

    transform_matricies, imgs = transform_matricies.to(device), imgs.to(device)

    # generate grid
    grid_indices, grid_cells, meshgrid, grid_grid = get_grid(gridsize[0], gridsize[1], gridsize[2],
                                                             points_distance=points_distance, info_size=4,
                                                             device=device)

    if len(grid_cells_path) > 0:
        grid_cells_data = torch.load(grid_cells_path)
        grid_cells = grid_cells_data["grid"].to(device)
        # color_difference = grid_cells_data["param"]["color_difference"].to(device)
        print(grid_cells.min(), grid_cells.max(), grid_cells.mean(), grid_cells.std())
        # grid_cells = grid_cells * (1/(grid_cells.mean()))
        # print(grid_cells.min(), grid_cells.max())
        grid_cells = grid_cells.clip(0.0, 1.0)
        if do_threshold:
            alphas = grid_cells[..., -1]
            grid_cells[..., -1][alphas < transparency_threshold] = 0.0
            # alphas[alphas > transparency_threshold] = 1.0
            # grid_cells[..., -1] = grid_cells[..., -1] * alphas

            # alphas = color_difference
            # grid_cells[..., -1][alphas < 0.5] = 0.0
            # alphas[alphas >= 0.40] = 1.0
            # grid_cells[..., -1] = grid_cells[..., -1] * alphas

    # choose an image to process
    img_index = imgindex  # 26#155 #45
    imgs = imgs[img_index, :, :, :].unsqueeze(0)
    transform_matricies = transform_matricies[img_index, :, :].unsqueeze(0)

    # generate samples
    t0 = time.time()
    samples_interval, pixels_to_rays, camera_positions, ray_directions = sample_camera_rays_batched(
        transform_matricies=transform_matricies,
        camera_angle_x=camera_angle_x,
        imgs=imgs,
        number_of_rays=number_of_rays,
        num_samples=num_samples,
        delta_step=delta_step,
        even_spread=even_spread,
        camera_ray=camera_ray,
        device=device
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
    pixels_color = pixels_color.cpu().detach().numpy()
    pixels_color = (pixels_color * 255).round().clip(0, 255).astype(np.uint8)
    image = pixels_color.reshape(200, 200, 4)
    image = np.transpose(image, (1, 0, 2))

    # -- sanity test for image --
    image_gt = pixels_to_rays.reshape(200, 200, 4).cpu().detach().numpy()
    image_gt = (image_gt * 255).astype(np.uint8)
    image_gt = np.transpose(image_gt, (1, 0, 2))

    print(time.time() - t0)

    # # plt.imshow(image)
    # # plt.imshow(image_gt)
    # # plt.show()
    #
    fig = plt.figure()

    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    fig.subplots_adjust(left=0.05, bottom=0.05, right=None, top=None, wspace=0.0, hspace=None)
    for ax in fig.axes:
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
    ax1.imshow(image)
    ax2.imshow(image_gt)
    fig.set_size_inches((8, 6))
    plt.show()

    np_array = grid_cells[..., -1].detach().flatten().cpu().numpy()
    plt.hist(np_array, bins=900)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Histogram of PyTorch Tensor Values')

    # Display the histogram
    plt.show()

    grid_cells = grid_cells.detach().cpu().numpy()

    # Get the indices of the non-zero depth values
    grid_cells_coords = np.argwhere(grid_cells[..., -1] > 0)

    # Extract the x, y, z coordinates and the RGB values
    x = grid_cells_coords[:, 0]
    y = grid_cells_coords[:, 1]
    z = grid_cells_coords[:, 2]
    rgb = grid_cells[x, y, z, :-1]
    d = grid_cells[x, y, z, -1]

    scatter = go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode='markers',
        marker=dict(
            size=2,
            color=rgb,
            opacity=1.0
        )
    )

    # Create the plot layout
    layout = go.Layout(
        title='RGBD Scatter3d Visualization',
        scene=dict(
            xaxis=dict(title='X Axis'),
            yaxis=dict(title='Y Axis'),
            zaxis=dict(title='Z Axis'),
        ),
    )

    # Create the plot figure
    fig = go.Figure(data=scatter, layout=layout)

    # Write the figure as an HTML file
    pio.write_html(fig, file='plotly_visualization.html', auto_open=True)

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
    # device = "cpu"
    # number_of_rays = 16
    # num_samples = 30#200
    # delta_step = 0.2
    # even_spread = True
    # camera_ray = False
    # points_distance = 0.5#*2*10
    # gridsize = [16, 16, 16]#64

    device = "cuda"
    number_of_rays = 25
    num_samples = 40  # 200
    delta_step = 0.05
    even_spread = False
    camera_ray = False
    points_distance = 0.05  # *2*10
    gridsize = [64, 64, 64]

    # device = "cpu"
    # number_of_rays = 16
    # num_samples = 30  # 200
    # delta_step = 0.2
    # even_spread = True
    # camera_ray = False
    # points_distance = 0.2  # *2*10
    # gridsize = [16, 16, 16]  # 64

    # device = "cuda"
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
                                                             points_distance=points_distance, info_size=4,
                                                             device=device)

    transform_matricies, file_paths, camera_angle_x = load_data(data)

    transform_matricies = transform_matricies.to(device)
    imgs = imgs.to(device)

    t0 = time.time()
    samples_interval, pixels_to_rays, camera_positions, ray_directions = sample_camera_rays_batched(
        transform_matricies=transform_matricies,
        camera_angle_x=camera_angle_x,
        imgs=imgs,
        number_of_rays=number_of_rays,
        num_samples=num_samples,
        delta_step=delta_step,
        even_spread=even_spread,
        camera_ray=camera_ray,
        device=device
    )
    normalized_samples_for_indecies = normalize_samples_for_indecies(grid_indices, samples_interval, points_distance)
    print(time.time() - t0)
    print(f"{samples_interval.shape=}\n{samples_interval[0]=}")
    print(f"{normalized_samples_for_indecies.shape=}\n{normalized_samples_for_indecies[0]=}")
    print(f"{normalized_samples_for_indecies.requires_grad=}")

    # -- visulize grid
    # ray_directions = ray_directions.to("cpu")
    # grid_indices = grid_indices.to("cpu")
    # visualize_rays_3d(ray_directions, [], grid_indices)

    # -- visulize camera rays and samples along rays
    ray_positions = torch.repeat_interleave(camera_positions, number_of_rays, 0)
    sampled_rays = samples_interval[num_samples*number_of_rays*10:num_samples*(number_of_rays*10 + number_of_rays)]
    ray_directions, ray_positions,  = ray_directions.to("cpu"), ray_positions.to("cpu")
    sampled_rays, grid_indices = sampled_rays.to("cpu"), grid_indices.to("cpu")

    # print(f"{grid_grid.shape=} {grid_grid}")

    downsample_factor = int(grid_grid.shape[0]/8)
    g = grid_grid[::downsample_factor, ::downsample_factor, ::downsample_factor, :]
    g = g.reshape(g.shape[0]*g.shape[1]*g.shape[2], 3).to("cpu")
    visualize_rays_3d(ray_directions, ray_positions, sampled_rays, g)
    # visualize_rays_3d(ray_directions, ray_positions, sampled_rays)


    # -- visulize grid around sampled points of ray
    # index = 5
    # selected_points, mask = collect_cell_information_via_indices(normalized_samples_for_indecies, meshgrid)
    # selected_points = selected_points.reshape([int(selected_points.shape[0] / 8), 8, 3])
    #
    # grid_grid = grid_grid.to("cpu")
    # selected_points = selected_points.to("cpu")
    # samples_interval = samples_interval.to("cpu")
    # ray_directions = ray_directions.to("cpu")
    #
    # paper_visulization(index, grid_grid, num_samples, number_of_rays, selected_points, samples_interval,
    #                    ray_directions)
    # result = trilinear_interpolation(normalized_samples_for_indecies, selected_points, grid_cells)

    # -- visulize points on grid closest to sampled points of ray
    # selected_points_voxels, outofbound_mask = get_nearest_voxels(normalized_samples_for_indecies, meshgrid)
    # grid_grid = grid_grid.to("cpu")
    # selected_points_voxels = selected_points_voxels.to("cpu")
    # samples_interval = samples_interval.to("cpu")
    # ray_directions = ray_directions.to("cpu")
    # print(f"{[t.is_cuda for t in [grid_grid, selected_points_voxels, samples_interval, ray_directions]]}")
    # voxel_visulization(index, grid_grid, num_samples, number_of_rays, selected_points_voxels,
    #                    samples_interval, ray_directions)

    # print(f"{outofbound_mask.shape=}\n{outofbound_mask.unique(return_counts =True)}")


if __name__ == "__main__":
    printi("start")
    fit()
    main()
    inference_test_voxels(grid_cells_path="grid_cells_trained.pth", transparency_threshold=0.1, imgindex=160,
                          do_threshold=True)
    printi("end")
