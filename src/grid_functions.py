import torch
import numpy as np


def trilinear_interpolation(normalized_samples_for_indecies, selected_points, grid_cells):
    """
    The input tensors:
        :param    normalized_samples_for_indecies has a shape of (N, 3), N 3D points normalized to cells indices.
        :param    selected_points has a shape (N, 8, 3), 8 corner points for each of the N 3D points.
        :param    grid_cells has a shape (X, Y, Z, cell information size), a 3D grid with XxYxZ cells,
        where each cell has a 3D value associated with it.
        :returns tensor of size [N, cell information size]

    selected_points are of this form:
        [['ceil_x', 'ceil_y', 'ceil_z'],
         ['ceil_x', 'ceil_y', 'floor_z'],
         ['ceil_x', 'floor_y', 'ceil_z'],
         ['ceil_x', 'floor_y', 'floor_z'],
         ['floor_x', 'ceil_y', 'ceil_z'],
         ['floor_x', 'ceil_y', 'floor_z'],
         ['floor_x', 'floor_y', 'ceil_z'],
         ['floor_x', 'floor_y', 'floor_z']]

    The final newstep tensor contains the interpolated values at the original input points based on the grid values.
    """

    inteplation_frac = torch.frac(normalized_samples_for_indecies)

    selection0 = grid_cells[selected_points[:, 4:, 0], selected_points[:, 4:, 1], selected_points[:, 4:, 2]]
    selection1 = grid_cells[selected_points[:, :4, 0], selected_points[:, :4, 1], selected_points[:, :4, 2]]

    inteplation_frac_step = inteplation_frac[:, 0].unsqueeze(-1).unsqueeze(-1)
    newstep = (selection1 * inteplation_frac_step) + (selection0 * (1 - inteplation_frac_step))
    del selection0, selection1

    inteplation_frac_step = inteplation_frac[:, 1].unsqueeze(-1).unsqueeze(-1)
    newstep = (newstep[:, :2] * inteplation_frac_step) + (newstep[:, 2:] * (1 - inteplation_frac_step))

    inteplation_frac_step = inteplation_frac[:, 2].unsqueeze(-1)
    newstep = (newstep[:, 0] * inteplation_frac_step) + (newstep[:, 1] * (1 - inteplation_frac_step))

    return newstep


def find_out_of_bound(indecies, grid):
    """
    Checks if the given indices are out of bounds of the given grid.
    :param indecies: A tensor of size (N, 3) containing the indices of the points to check.
    :param grid: A tensor representing the grid in which the points are located.
    :return: A boolean tensor of size (N,) where False indicates the corresponding index is out of bounds.
    """
    idx1, idx2, idx3 = indecies[:, 0], indecies[:, 1], indecies[:, 2]

    # find which points are outside the grid
    X, Y, Z, N = grid.shape
    idx1_outofbounds = (idx1 < X) & (idx1 >= 0)
    idx2_outofbounds = (idx2 < Y) & (idx2 >= 0)
    idx3_outofbounds = (idx3 < Z) & (idx3 >= 0)
    outofbounds = idx1_outofbounds & idx2_outofbounds & idx3_outofbounds

    return outofbounds


def fix_out_of_bounds(indices, grid):
    """
    Fixes out of bounds indices by applying periodic boundary conditions.
    :param indices: A tensor of size (N, 3) containing the coordinates of the points.
    :param grid: A tensor representing the grid in which the points are located.
    :return: Three tensors of size (N,) representing the fixed indices in x, y, and z directions.
    """
    idx1, idx2, idx3 = indices[:, 0], indices[:, 1], indices[:, 2]
    X, Y, Z, N = grid.shape
    idx1 %= X
    idx2 %= Y
    idx3 %= Z

    return idx1, idx2, idx3


def get_nearest_voxels(normalized_samples_for_indices, grid):
    """
    Gets the nearest voxel values for the given normalized samples in the grid.
    :param normalized_samples_for_indices: A tensor of size (N, 3) containing normalized sample points.
    :param grid: A tensor representing the grid in which the points are located.
    :return: A tuple containing the tensor with the nearest voxel values and a boolean tensor indicating
    """
    indices = torch.round(normalized_samples_for_indices).to(torch.long)
    outofbounds = find_out_of_bound(indices, grid)
    idx1, idx2, idx3 = fix_out_of_bounds(indices, grid)
    return grid[idx1, idx2, idx3], outofbounds


def collect_cell_information_via_indices(normalized_samples_for_indices, B):
    """
    Collects cell information for the given indices in the grid.
    :param normalized_samples_for_indices: A tensor of size (N, 3) containing the normalized sample points.
    :param B: A tensor representing the grid in which the points are located.
    :return: A tuple containing the tensor with the collected cell information and a boolean tensor indicating
    out-of-bounds indices.
    """
    outofbounds = find_out_of_bound(normalized_samples_for_indices, B)
    edge_matching_points = get_grid_points_indices(normalized_samples_for_indices)
    A = edge_matching_points.reshape([edge_matching_points.shape[0] * edge_matching_points.shape[1],
                                      edge_matching_points.shape[2]])

    # Use advanced indexing to get the values
    idx1, idx2, idx3 = fix_out_of_bounds(A, B)
    output = B[idx1, idx2, idx3]
    return output, outofbounds



def generate_grid(sx, sy, sz, points_distance=0.5, info_size=4, device="cuda"):
    """
    Generates a grid with the specified size, point distance, and information size.
    :param sx: Number of grid points in the x direction.
    :param sy: Number of grid points in the y direction.
    :param sz: Number of grid points in the z direction.
    :param points_distance: Distance between grid points (default=0.5).
    :param info_size: Number of additional information channels for each grid cell (default=4).
    :param device: Device on which the tensors should be created (default="cuda").
    :return: A tuple containing the grid coordinates,
        grid cells: a 3d tensor containing information within the grid
        meshgrid: normalized coordinates of the grid for use as indices
        grid_grid: coordinates of the grid in image space
    """
    grindx_indices, grindy_indices, grindz_indices = torch.arange(sx, device=device), torch.arange(sy, device=device), \
        torch.arange(sz, device=device)
    coordsx, coordsy, coordsz = torch.meshgrid(grindx_indices, grindy_indices, grindz_indices, indexing='ij')

    meshgrid = torch.stack([coordsx, coordsy, coordsz], dim=-1)

    # center grid
    coordsx, coordsy, coordsz = coordsx - np.ceil(sx / 2) + 1, coordsy - np.ceil(sy / 2) + 1, coordsz - np.ceil(
        sz / 2) + 1

    # edit grid spacing
    coordsx, coordsy, coordsz = coordsx * points_distance, coordsy * points_distance, coordsz * points_distance

    grid_grid = torch.stack([coordsx, coordsy, coordsz], dim=-1)
    grid_coords = grid_grid.reshape(sx * sy * sz, 3)

    grid_cells = torch.zeros([grid_grid.shape[0], grid_grid.shape[1], grid_grid.shape[2], info_size],
                             requires_grad=True, device=device)

    return grid_coords, grid_cells, meshgrid, grid_grid


def get_grid_points_indices(normalized_samples_for_indecies):
    """
    calculates the indices of grid points surrounding each input point. It returns the indices of the 8 corners of the
    grid cell that encloses each input point.

    :param normalized_samples_for_indecies tensor of size (N, 3):
    :returns A tensor of shape (N, 8, 3) containing the indices of the 8 corners of the grid cell that encloses each
    input point.
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
