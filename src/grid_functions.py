import torch
import numpy as np


def trilinear_interpolation(normalized_samples_for_indecies, selected_points, grid_cells):
    """
    The input tensors:
        :param    normalized_samples_for_indecies has a shape of (N, 3), which represents N 3D points.
        :param    selected_points has a shape (N, 8, 3), which represents 8 corner points for each of the N 3D points.
        :param    grid_cells has a shape (X, Y, Z, 3), which represents a 3D grid with XxYxZ cells,
        where each cell has a 3D value associated with it.

    selected_points are of this form:
        [['ceil_x', 'ceil_y', 'ceil_z'],
         ['ceil_x', 'ceil_y', 'floor_z'],
         ['ceil_x', 'floor_y', 'ceil_z'],
         ['ceil_x', 'floor_y', 'floor_z'],
         ['floor_x', 'ceil_y', 'ceil_z'],
         ['floor_x', 'ceil_y', 'floor_z'],
         ['floor_x', 'floor_y', 'ceil_z'],
         ['floor_x', 'floor_y', 'floor_z']]

    1. selection0 and selection1:
        These two tensors are created by reshaping the last 4 and first 4 corner points of each cell in selected_points,
        respectively.
        Then, they extract the values from grid_cells corresponding to these corner points.

    2. interpolation_frac_step1:
        This tensor represents the interpolation fraction for the first dimension.
        The code multiplies selection1 by this fraction and selection0 by its complement (1 - fraction) and adds them
        together, yielding newstep.

    3. inteplation_frac_step2:
        This tensor represents the interpolation fraction for the second dimension.
        The code performs interpolation for the second dimension using the values computed in step 3 and updates
        newstep.

    4. inteplation_frac_step3:
        This tensor represents the interpolation fraction for the third dimension.
        The code performs interpolation for the third dimension using the values computed in step 4 and updates newstep.

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
    idx1, idx2, idx3 = indecies[:, 0], indecies[:, 1], indecies[:, 2]

    # find which points are outside the grid
    X, Y, Z, N = grid.shape
    idx1_outofbounds = (idx1 < X) & (idx1 >= 0)
    idx2_outofbounds = (idx2 < Y) & (idx2 >= 0)
    idx3_outofbounds = (idx3 < Z) & (idx3 >= 0)
    outofbounds = idx1_outofbounds & idx2_outofbounds & idx3_outofbounds

    return outofbounds


def fix_out_of_bounds(indecies, grid):
    idx1, idx2, idx3 = indecies[:, 0], indecies[:, 1], indecies[:, 2]
    X, Y, Z, N = grid.shape
    idx1 %= X
    idx2 %= Y
    idx3 %= Z

    return idx1, idx2, idx3


def get_nearest_voxels(normalized_samples_for_indecies, grid):
    indecies = torch.round(normalized_samples_for_indecies).to(torch.long)
    outofbounds = find_out_of_bound(indecies, grid)
    idx1, idx2, idx3 = fix_out_of_bounds(indecies, grid)
    return grid[idx1, idx2, idx3], outofbounds


def collect_cell_information_via_indices(normalized_samples_for_indecies, B):
    outofbounds = find_out_of_bound(normalized_samples_for_indecies, B)
    edge_matching_points = get_grid_points_indices(normalized_samples_for_indecies)
    A = edge_matching_points.reshape([edge_matching_points.shape[0] * edge_matching_points.shape[1],
                                      edge_matching_points.shape[2]])

    # Use advanced indexing to get the values
    idx1, idx2, idx3 = fix_out_of_bounds(A, B)
    output = B[idx1, idx2, idx3]
    return output, outofbounds


def get_grid_points_indices(normalized_samples_for_indecies):
    """
    Given grid indices, a samples interval, and a points distance, this function calculates the indices of
    grid points surrounding each input point. It returns the indices of the 8 corners of the grid cell
    that encloses each input point.

    Args:
            :param normalized_samples_for_indecies:
    Returns:
        torch.Tensor: A tensor of shape (N, 8, 3) containing the indices of the 8 corners of the grid cell
                      that encloses each input point.
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


def get_grid(sx, sy, sz, points_distance=0.5, info_size=4, device="cuda"):
    grindx_indices, grindy_indices, grindz_indices = torch.arange(sx, device=device), torch.arange(sy, device=device), \
        torch.arange(sz, device=device)
    coordsx, coordsy, coordsz = torch.meshgrid(grindx_indices, grindy_indices, grindz_indices, indexing='ij')

    meshgrid = torch.stack([coordsx, coordsy, coordsz], dim=-1)

    # center grid
    coordsx, coordsy, coordsz = coordsx - np.ceil(sx / 2) + 1, coordsy - np.ceil(sy / 2) + 1, coordsz - np.ceil(
        sz / 2) + 1

    # edit grid spacing
    coordsx, coordsy, coordsz = coordsx * points_distance, coordsy * points_distance, coordsz * points_distance

    # make it so no points of the grid are underground (currently off, the assumption is untrue)
    # coordsz = coordsz - coordsz.min()

    grid_grid = torch.stack([coordsx, coordsy, coordsz], dim=-1)
    grid_coords = grid_grid.reshape(sx * sy * sz, 3)

    grid_cells = torch.zeros([grid_grid.shape[0], grid_grid.shape[1], grid_grid.shape[2], info_size],
                             requires_grad=True, device=device)

    return grid_coords, grid_cells, meshgrid, grid_grid
