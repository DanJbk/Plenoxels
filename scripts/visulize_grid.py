import plotly.graph_objs as go
import plotly.io as pio
import numpy as np
import argparse
import torch

from matplotlib import pyplot as plt
from src.grid_functions import convolve_grid_to_remove_noise


def main(args):
    grid_cells_path = args.grid_cells_path
    do_threshold = args.do_threshold
    threshold = args.threshold
    savepath = args.savepath
    clean_strays = args.clean_strays

    visulize_grid_ploty(grid_cells_path, threshold, do_threshold, clean_strays, savepath)


def visulize_grid_ploty(grid_cells_path, threshold, do_threshold, clean_strays, savepath=None, visulize_colors_grad=False):

    grid_cells_data = torch.load(grid_cells_path)
    grid_cells = grid_cells_data["grid"]

    if 'grid_grad' in grid_cells_data.keys():
        grid_cells_grad = grid_cells_data['grid_grad'].detach().cpu()

        if do_threshold:
            grid_cells[:, :, :, -1][(torch.norm(grid_cells_grad[..., :-1], dim=-1) < 3.0e-06)] = 0

    # ---

    if do_threshold:
        grid_cells = grid_cells.clip(0.0, 1.0)
        alphas = grid_cells[..., -1]
        grid_cells[..., -1][alphas < threshold] = 0.0
        grid_cells[..., -1][alphas > 0] = 1.

    if clean_strays == True:
        grid_cells = convolve_grid_to_remove_noise(grid_cells, kernel_size=3, radius=1.0, threshold=2, repeats=20)
        grid_cells = convolve_grid_to_remove_noise(grid_cells, kernel_size=7, radius=3.0, threshold=4, repeats=20)

    # ---

    grid_cells = grid_cells.detach().cpu().numpy()

    # Get the indices of the larger-than-zero opacity values
    grid_cells_coords = np.argwhere(grid_cells[..., -1] > 0)

    # Extract the x, y, z coordinates and the RGB values
    x = grid_cells_coords[:, 0]
    y = grid_cells_coords[:, 1]
    z = grid_cells_coords[:, 2]
    rgb = grid_cells[x, y, z, :-1]

    if visulize_colors_grad:
        grid_cells_coords = np.argwhere(
            np.logical_and(
                torch.norm(grid_cells_grad[..., :], dim=-1).numpy() < 2.0e-4, # 4.0e-06,
                grid_cells[..., -1] > 0
            )
        )

        x, y, z = grid_cells_coords[:, 0], grid_cells_coords[:, 1], grid_cells_coords[:, 2]
        rgb = (torch.norm(grid_cells_grad[..., :-1], dim=-1).numpy()[x,y,z]**0.42)

    # Write the figure as an HTML file
    if savepath:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x, y, z, c=rgb)
        plt.savefig(savepath, dpi=300)
        return True

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
    pio.write_html(fig, file='plotly_visualization.html', auto_open=True)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='visualize camera positions around a grid.')
    parser.add_argument('--threshold', default=0.0, type=float, help='opacity threshold for showing the voxel')
    parser.add_argument('--do_threshold', default=True, type=bool, help='opacity threshold for showing the voxel')
    parser.add_argument('--grid_cells_path', default="src/grid_cells_trained.pth", help='path to saved tensor')
    parser.add_argument('--savepath', default=None, help='path to saved tensor')
    parser.add_argument('--clean_strays', default=False, help='clean stray pixels')
    args = parser.parse_args()

    main(args)
