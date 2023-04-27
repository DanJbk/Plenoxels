
import torch

from scripts.compare_inference_to_image import compare_grid_to_image
from scripts.train import fit
from scripts.visulize_camera_and_grid import view_grid_cameras
from scripts.visulize_grid import visulize_grid_ploty


def main():

    # arguments

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    save_path = "src/grid_cells_trained.pth"

    train_path = "data/lego/train"
    test_path = "data/lego/test"

    transform_path_train = "data/lego/transforms_train.json"
    transform_path_test = "data/lego/transforms_test.json"

    gridsize = [256, 256, 256]
    points_distance = 0.0125

    number_of_rays = 128
    num_samples = 600
    delta_step = 0.0125

    lr = 0.0075
    steps = 500

    tv = 2.5
    beta = 0.005

    even_spread = False

    # ---

    # view cameras in relation to grid
    view_grid_cameras(train_path, transform_path_train, gridsize, points_distance, device)

    # train
    fit(gridsize, points_distance, number_of_rays, num_samples, delta_step, lr, tv, beta, steps, even_spread,
        train_path,
        transform_path_train,
        save_path, device)

    # visualize result in 2d
    compare_grid_to_image(test_path, transform_path_test, save_path, 26, do_threshold=True, transparency_threshold=0.1,
                          number_of_rays=10000, num_samples=num_samples, device=device)

    # visualize result in 3d
    visulize_grid_ploty(save_path, threshold=0.1, do_threshold=True)


if __name__ == "__main__":
    main()
