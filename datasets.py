import numpy as np
import torch

class GaussianTask:
    """
    A simple data generator for creating data from multidimensional Gaussians with different means and same scale
    """

    def __init__(self, emb_dim, n_gaussians, locs, scales, labels=None):
        self.emb_dim = emb_dim
        self.n_gaussians = n_gaussians
        self.locs = np.array(locs)
        self.scales = scales
        self.labels = labels if labels is not None else list(range(n_gaussians))

    def create(self, n_samples):
        x = []
        y = []
        for i in range(self.n_gaussians):
            x.append(torch.randn(n_samples, self.emb_dim) * self.scales[i] + self.locs[i])
            y.append(torch.ones(n_samples) * self.labels[i])

        return torch.vstack(x), torch.hstack(y)[:, None]

    def project_data(self, x):
        if self.emb_dim == 1:
            return x
        proj_vec = np.repeat(self.locs[1], self.emb_dim).astype(float) - np.repeat(self.locs[0], self.emb_dim).astype(
            float)
        proj_vec /= np.linalg.norm(proj_vec).astype(float)  # get normalized vector for projection
        return x @ proj_vec[:, None]

    def get_centers_grid(self, n_samples):
        alphas = np.linspace(0, 1, n_samples)
        loc0 = self.locs[0][None] - 3 * self.scales[0]
        loc1 = self.locs[1][None] + 3 * self.scales[1]
        dist = loc1 - loc0
        grid_x = loc0 + alphas[:, None] * dist
        grid_x = np.tile(grid_x, (1, self.n_gaussians))
        return grid_x
