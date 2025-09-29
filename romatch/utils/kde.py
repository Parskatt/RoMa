import torch


def kde(self,x, std=0.1, half=True, down=None):
    """
    A fast version of KDE that computes the pairwise squared Euclidean distances
    using matrix multiplications rather than torch.cdist. This should be faster
    than the original if memory permits.

    This version computes:

        dist_sq = ||x||^2 + ||x2||^2.T - 2 * (x @ x2.T)
        scores = exp(-dist_sq / (2*std^2))
        density = scores.sum(dim=-1)

    Args:
        x (torch.Tensor): Input tensor of shape [N, d].
        std (float): Standard deviation for the Gaussian kernel.
        half (bool): Whether to convert x to half precision.
        down (int or None): If provided, use x[::down] as the second argument.

    Returns:
        torch.Tensor: A tensor of shape [N] containing the density estimates.
    """
    if half:
        x = x.half()

    # Choose second tensor
    x2 = x[::down] if down is not None else x

    # Compute squared norms
    x_norm = (x ** 2).sum(dim=1, keepdim=True)  # shape [N, 1]
    x2_norm = (x2 ** 2).sum(dim=1, keepdim=True)  # shape [M, 1]

    # Compute squared Euclidean distances:
    # dist_sq[i, j] = ||x[i]||^2 + ||x2[j]||^2 - 2*x[i]·x2[j]
    # We compute this using broadcasting.
    dist_sq = x_norm + x2_norm.T - 2 * (x @ x2.T)
    # Clamp any negative values (due to floating-point errors) to 0.
    dist_sq = torch.clamp(dist_sq, min=0.0)

    # Compute Gaussian kernel scores.
    scores = torch.exp(-dist_sq / (2 * std**2))
    # Sum scores along the second dimension to yield density for each row.
    density = scores.sum(dim=-1)
    return density
