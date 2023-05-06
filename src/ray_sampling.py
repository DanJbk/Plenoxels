import torch
import numpy as np

from src.data_processing import get_data_from_index
from src.utils import tensor_linspace, batched_cartesian_prod


def get_camera_normal(transform_matrix):
    fixed_pos = transform_matrix[:3, :3].unsqueeze(0)
    fixed_pos[:, 2] = fixed_pos[:, 2] * -1
    return fixed_pos


def normalize_samples_for_indecies(grid_indices, samples_interval, points_distance):
    return (samples_interval - grid_indices.min(0)[0]) / points_distance


def sample_positive_opacity(imgs, number_of_rays, device):
    """
    :param imgs: tensor of shape [Cameras, H, W, 4]
    :param number_of_rays: int
    :param device: "cpu"/"cuda"
    :return: normalized indices of the image with none-zero opacity
    """

    none_zero_opacity_indices = imgs[..., -1].nonzero(as_tuple=False)
    ranges = torch.cat([torch.tensor([0], device=device), torch.diff(none_zero_opacity_indices[:, 0]).nonzero().squeeze(),
                        torch.tensor([none_zero_opacity_indices.shape[0]], device=device)])
    ranges_diff = ranges[1:] - ranges[:-1]
    lower_bounds = ranges[:-1].unsqueeze(1)

    result = (lower_bounds + (
                torch.rand(ranges_diff.size(0), number_of_rays, device=device) * ranges_diff.unsqueeze(1))).type(
        torch.long).flatten()
    ray_indices = none_zero_opacity_indices[result][:, 1:].reshape(imgs.shape[0], number_of_rays, 2)
    ray_indices[:, :, 0] = ray_indices[:, :, 0]/imgs.shape[1]
    ray_indices[:, :, 1] = ray_indices[:, :, 1]/imgs.shape[2]

    return ray_indices


def generate_rays(num_rays, transform_matrix, camera_angle_x, even_spread=False):
    """
    Generates rays originating from the camera with the specified number of rays, transform matrix, and camera angle.
    :param num_rays: Number of rays to generate.
    :param transform_matrix: The 4x4 camera transformation matrix.
    :param camera_angle_x: The camera's horizontal field of view angle.
    :param even_spread: If True, generates rays with an even spread. If False, generates random rays.
    :return: A tensor of shape (num_rays, 3) containing the generated ray directions.
    """


    # Extract camera axes and position
    camera_x_axis = transform_matrix[:3, 0]
    camera_y_axis = transform_matrix[:3, 1]
    camera_z_axis = -transform_matrix[:3, 2]

    # Compute the aspect ratio (width / height) of the camera's FOV
    aspect_ratio = camera_x_axis.norm() / camera_y_axis.norm()

    # Generate evenly spread ray indices
    if even_spread:
        num_rays_sqrt = np.round(np.sqrt(num_rays))

        u_values = torch.linspace(-0.5 * camera_angle_x, 0.5 * camera_angle_x, int(num_rays_sqrt))
        v_values = torch.linspace(-0.5 * camera_angle_x / aspect_ratio, 0.5 * camera_angle_x / aspect_ratio,
                                  int(num_rays_sqrt))
        ray_indices = torch.cartesian_prod(u_values, v_values)

    else:
        # Generate random ray indices
        ray_indices = torch.rand(num_rays, 2)  # Generate random numbers in the range [0, 1)

        # Scale and shift the ray indices to match the camera's FOV
        ray_indices[:, 0] = camera_angle_x * (ray_indices[:, 0] - 0.5)
        ray_indices[:, 1] = camera_angle_x / aspect_ratio * (ray_indices[:, 1] - 0.5)

    # Get the u and v values from ray_indices
    u_values, v_values = ray_indices[:, 0].unsqueeze(-1), ray_indices[:, 1].unsqueeze(-1)
    directions = u_values * camera_x_axis + v_values * camera_y_axis + camera_z_axis

    ray_directions = directions / directions.norm(dim=1).unsqueeze(-1)

    return ray_directions


def sample_camera_rays(data, number_of_rays, num_samples, delta_step, even_spread, camera_ray):
    """
    Samples camera rays based on the input data.
    :param data: The input data containing camera information.
    :param number_of_rays: Number of rays to generate for each camera.
    :param num_samples: Number of samples to take along each ray.
    :param delta_step: The step size along the ray.
    :param even_spread: If True, generates rays with an even spread. If False, generates random rays.
    :param camera_ray: If True, generates rays directly opposite the camera's forward direction.
    :return: A tuple containing the tensors for samples interval, camera positions, and ray directions.
    """
    if even_spread:
        number_of_rays = int(np.round(np.sqrt(number_of_rays)) ** 2)

    # Preallocate memory for ray_directions and camera_positions
    num_frames = len(data["frames"])
    ray_directions = torch.zeros((num_frames * number_of_rays, 3)) if not camera_ray else torch.zeros((num_frames, 3))
    camera_positions = torch.zeros((num_frames, 3))

    for i in range(num_frames):
        transform_matrix, rotation, file_path, camera_angle_x = get_data_from_index(data, i)

        if camera_ray:
            current_ray_directions = transform_matrix[:3, 2].unsqueeze(0) * -1
            ray_directions[i] = current_ray_directions
        else:
            current_ray_directions = generate_rays(number_of_rays, transform_matrix, camera_angle_x,
                                                   even_spread=even_spread)
            ray_directions[i * number_of_rays:(i + 1) * number_of_rays] = current_ray_directions

        camera_positions[i] = transform_matrix[:3, 3]

    delta_forsamples = delta_step * torch.arange(num_samples + 1)[1:].repeat(number_of_rays * len(data["frames"])) \
        .unsqueeze(1)

    camera_positions_forsamples = torch.repeat_interleave(camera_positions, num_samples * number_of_rays, 0)

    ray_directions_forsamples = torch.repeat_interleave(ray_directions, num_samples, 0)
    samples_interval = camera_positions_forsamples + ray_directions_forsamples * delta_forsamples

    return samples_interval, camera_positions, ray_directions


def sample_camera_rays_batched(transform_matrices, camera_angle_x, imgs, number_of_rays, num_samples, delta_step,
                               even_spread, camera_ray, device='cuda'):
    """
    Samples camera rays in a batched manner based on the input data.
    :param transform_matrices: a tensor of Camera_numberx4x4 camera transformation matrices.
    :param camera_angle_x: The camera's horizontal field of view angle.
    :param imgs: Batch of images corresponding to the input data.
    :param number_of_rays: Number of rays to generate for each camera.
    :param num_samples: Number of samples to take along each ray.
    :param delta_step: The step size along the ray.
    :param even_spread: If True, generates rays with an even spread. If False, generates random rays.
    :param camera_ray: If True, generates rays directly opposite the camera's forward direction.
    :param device: Device on which the tensors should be created (default='cuda').
    :return: A tuple containing the tensors for samples interval, pixels to rays mapping, camera positions,
    and current ray directions.
    """
    if even_spread:
        number_of_rays = int(np.round(np.sqrt(number_of_rays)) ** 2)

    # transform_matricies, file_paths, camera_angle_x = load_data(data)

    pixels_to_rays = []
    if camera_ray:
        current_ray_directions = transform_matrices[:, :3, 2].unsqueeze(0) * -1

    else:
        current_ray_directions, pixels_to_rays = generate_rays_batched(imgs, number_of_rays, transform_matrices,
                                                                       camera_angle_x,
                                                                       even_spread=even_spread, device=device)

    # Move tensors to specified device
    current_ray_directions = current_ray_directions
    camera_positions = transform_matrices[:, :3, 3]

    delta_forsamples = delta_step * torch.arange(num_samples + 1, device=device)[1:].repeat(number_of_rays *
                                                                                            imgs.shape[0]).unsqueeze(1)

    camera_positions_forsamples = torch.repeat_interleave(camera_positions, num_samples * number_of_rays, 0)

    ray_directions_for_samples = torch.repeat_interleave(current_ray_directions, num_samples, 0)
    samples_interval = camera_positions_forsamples + ray_directions_for_samples * delta_forsamples

    return samples_interval, pixels_to_rays, camera_positions, current_ray_directions


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


def generate_rays_batched(imgs, number_of_rays, transform_matricies, camera_angle_x, even_spread=False, device='cuda'):
    """
        Generates rays for each camera and corresponding pixel indices.

        Args:
            imgs (torch.Tensor): Batch of input images (B, H, W).
            number_of_rays (int): Number of rays to generate for each camera.
            transform_matricies (torch.Tensor): Transformation matrices for cameras (B, 4, 4).
            camera_angle_x (float): Horizontal field of view of the camera.
            device (str, optional): Device to use for computation. Defaults to 'cpu'.
            even_spread (bool): Whether to generate rays with even spacing. Defaults to False (random spacing).

        Returns:
            torch.Tensor: Generated ray directions (B * number_of_rays, 3).
            torch.Tensor: Pixel colors corresponding to the generated rays.
        """

    num_cameras = transform_matricies.shape[0]

    # Move tensors to the specified device
    # imgs = imgs.to(device)
    # transform_matricies = transform_matricies.to(device)

    # Extract camera axes from transformation matrices
    camera_x_axis = transform_matricies[:, :3, 0]
    camera_y_axis = transform_matricies[:, :3, 1]
    camera_z_axis = -transform_matricies[:, :3, 2]
    aspect_ratio = camera_x_axis.norm(dim=1) / camera_y_axis.norm(dim=1)

    if even_spread:
        num_rays_sqrt = np.round(np.sqrt(number_of_rays))

        # Compute evenly spaced u and v values for rays
        start = torch.tensor([-0.5 * camera_angle_x], device=device).repeat(aspect_ratio.shape)
        u_values = torch.linspace(-0.5 * camera_angle_x, 0.5 * camera_angle_x, int(num_rays_sqrt),
                                  device=device).unsqueeze(0).repeat([aspect_ratio.shape[0], 1])
        v_values = -tensor_linspace(start / aspect_ratio, -start / aspect_ratio, int(num_rays_sqrt), device=device)

        # Calculate ray indices using Cartesian product
        ray_indices = batched_cartesian_prod(u_values, v_values)

        # normalize to image space
        pixel_indices = ray_indices.clone()
        pixel_indices[:, :, 0] = (pixel_indices[:, :, 0] / camera_angle_x) + 0.5
        pixel_indices[:, :, 1] = -(pixel_indices[:, :, 1] / ((camera_angle_x * (1 / aspect_ratio)).unsqueeze(1))) + 0.5

    else:
        # Generate random ray indices
        ray_indices = torch.rand(transform_matricies.shape[0], number_of_rays,
                                 2, device=device)  # Generate random numbers in the range [0, 1)

        pixel_indices = ray_indices.clone()

        # Scale and shift the ray indices to match the camera's FOV
        ray_indices[:, :, 0] = camera_angle_x * (ray_indices[:, :, 0] - 0.5)
        ray_indices[:, :, 1] = -((camera_angle_x * (1 / aspect_ratio)).unsqueeze(1) * (ray_indices[:, :, 1] - 0.5))

    # Clamp pixel indices to image dimensions
    pixel_indices[:, :, 0] = (imgs.shape[1] * pixel_indices[:, :, 0]).round().clamp(max=imgs.shape[1] - 1)
    pixel_indices[:, :, 1] = (imgs.shape[2] * pixel_indices[:, :, 1]).round().clamp(max=imgs.shape[2] - 1)
    pixel_indices = pixel_indices.to(torch.long)

    # Map pixel indices to camera indices
    camera_to_ray = torch.repeat_interleave(torch.arange(0, transform_matricies.shape[0], device=device),
                                            number_of_rays, 0)
    pixel_indices = pixel_indices.reshape([pixel_indices.shape[0] * pixel_indices.shape[1], pixel_indices.shape[2]])
    # pixels_to_rays = imgs[camera_to_ray, pixel_indices[:, 0], pixel_indices[:, 1]]
    # the y dimension is the first one, the x dimension is the second in an image
    pixels_to_rays = imgs[camera_to_ray, pixel_indices[:, 1], pixel_indices[:, 0]]

    # Get the u and v values from ray_indices
    u_values, v_values = ray_indices[:, :, 0].unsqueeze(-1), ray_indices[:, :, 1].unsqueeze(-1)
    u_values = u_values.expand(num_cameras, number_of_rays, 3)
    v_values = v_values.expand(num_cameras, number_of_rays, 3)

    # Expand camera axes for broadcasting
    camera_x_axis = camera_x_axis.unsqueeze(1).expand(num_cameras, number_of_rays, 3)
    camera_y_axis = camera_y_axis.unsqueeze(1).expand(num_cameras, number_of_rays, 3)
    camera_z_axis = camera_z_axis.unsqueeze(1).expand(num_cameras, number_of_rays, 3)

    # Compute ray directions and normalize
    directions = u_values * camera_x_axis + v_values * camera_y_axis + camera_z_axis
    ray_directions = directions / directions.norm(dim=2).unsqueeze(-1)

    return ray_directions.reshape(ray_directions.shape[0] * number_of_rays, -1), pixels_to_rays



