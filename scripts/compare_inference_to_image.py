import argparse

from matplotlib import pyplot as plt
import numpy as np
import torch

from src.ray_sampling import normalize_samples_for_indecies, sample_camera_rays_batched
from src.data_processing import load_data, load_image_data_from_path
from src.grid_functions import get_nearest_voxels, generate_grid
from src.rays_logic import compute_alpha_weighted_pixels
from src.visualization import visulize_3d_in_2d_fast


def main(args):

    device = args.device
    if device == "auto":
        device = "cpu" if torch.cuda.is_available() else "cuda:0"

    transparency_threshold = args.threshold
    imgindex = args.image_num
    do_threshold = args.do_threshold
    number_of_rays = args.ray_num
    num_samples = args.sample_num  # 200
    path = args.path
    transform_path = args.transform_path
    grid_cells_path = args.grid_cells_path

    compare_grid_to_image(path, transform_path, grid_cells_path, imgindex, do_threshold, transparency_threshold,
                          number_of_rays, num_samples, device)


def compare_grid_to_image(path, transform_path, grid_cells_path, imgindex, do_threshold, transparency_threshold,
                          number_of_rays, num_samples, device, save_image=False, image_path=None):

    data, imgs = load_image_data_from_path(path, transform_path)
    transform_matrices, file_paths, camera_angle_x = load_data(data)

    transform_matrices, imgs = transform_matrices.to(device), imgs.to(device)

    grid_cells_data = torch.load(grid_cells_path)
    grid_cells = grid_cells_data["grid"].detach().to(device)
    grid_cells = grid_cells.clip(0.0, 1.0)
    if do_threshold:
        alphas = grid_cells[..., -1]
        grid_cells[..., -1][alphas < transparency_threshold] = 0.0

    gridsize = grid_cells.shape
    points_distance = grid_cells_data["param"]["points_distance"]
    delta_step = grid_cells_data["param"]["delta_step"]

    # generate grid
    grid_indices, _, _, _ = generate_grid(gridsize[0], gridsize[1], gridsize[2],
                                          points_distance=points_distance, info_size=4,
                                          device=device)
    # choose an image to process
    imgs = imgs[imgindex, :, :, :].unsqueeze(0)
    image = visulize_3d_in_2d_fast(grid_cells, points_distance, transform_matrices[imgindex], camera_angle_x,
                                   size_y=500)

    # image ground truth
    image_gt = (imgs * 255).squeeze().detach().cpu().numpy().astype(np.uint8)

    if image_path and save_image:
        # fig.savefig(image_path)
        plt.imsave(image_path, np.ascontiguousarray(image))
        return

    # display
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


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='visualize camera positions around a grid.')
    parser.add_argument('--device', default="auto", help='device to run on (cpu, cuda, auto)')
    parser.add_argument('--threshold', default=0.2, type=float, help='opacity threshold for showing the voxel')
    parser.add_argument('--do_threshold', default=True, type=bool, help='opacity threshold for showing the voxel')
    parser.add_argument('--point_distance', default=0.05, type=float, help='distance from point to point')
    parser.add_argument('--grid_cells_path', default="src/grid_cells_trained_2000_steps_quality_parameters.pth", help='path to dataset')
    parser.add_argument('--transform_path', default="data/ship/transforms_train.json", help='path to json file')
    parser.add_argument('--image_num', default=0, type=int, help='index of image to view')
    parser.add_argument('--ray_num', default=40000, type=int, help='resolution of image, should (width/height) squared')
    parser.add_argument('--sample_num', default=600, type=int, help='number of samples along a ray')
    parser.add_argument('--path', default="data/ship/train", help='path to dataset')
    args = parser.parse_args()

    main(args)
