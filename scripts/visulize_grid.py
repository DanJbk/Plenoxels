
import plotly.graph_objs as go
import plotly.io as pio
import numpy as np
import torch
import argparse


def main(args):
    grid_cells_path = args.grid_cells_path
    do_threshold = args.do_threshold
    threshold = args.threshold

    visulize_grid_ploty(grid_cells_path, threshold, do_threshold)


def visulize_grid_ploty(grid_cells_path, threshold, do_threshold):

    grid_cells_data = torch.load(grid_cells_path)
    grid_cells = grid_cells_data["grid"]

    grid_cells = grid_cells.clip(0.0, 1.0)
    if do_threshold:
        alphas = grid_cells[..., -1]
        grid_cells[..., -1][alphas < threshold] = 0.0

    grid_cells = grid_cells.detach().cpu().numpy()

    # Get the indices of the non-zero depth values
    grid_cells_coords = np.argwhere(grid_cells[..., -1] > 0)

    # Extract the x, y, z coordinates and the RGB values
    x = grid_cells_coords[:, 0]
    y = grid_cells_coords[:, 1]
    z = grid_cells_coords[:, 2]
    rgb = grid_cells[x, y, z, :-1]

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


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='visualize camera positions around a grid.')
    parser.add_argument('--threshold', default=0.2, type=float, help='opacity threshold for showing the voxel')
    parser.add_argument('--do_threshold', default=True, type=bool, help='opacity threshold for showing the voxel')
    parser.add_argument('--grid_cells_path', default="src/grid_cells_trained.pth", help='path to saved tensor')
    args = parser.parse_args()

    main(args)
