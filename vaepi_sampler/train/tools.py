import numpy as np
import torch


def sample_z_input(batch_size, latent_dim) -> torch.Tensor:

    "samples the latent variable at the input of the (reverse) VAE"

    return torch.randn((batch_size, latent_dim))


def minimal_path_SHO(N_T, T_max, x0, xf):

    "returns the minimal Euclidean path for an SHO"

    x = [x0]
    dt = T_max / (N_T - 1)

    for i in range(1, N_T):
        x.append(
            x0 * np.cosh(i * dt)
            + (xf - x0 * np.cosh(T_max)) / np.sinh(T_max) * np.sinh(i * dt)
        )

    return x
