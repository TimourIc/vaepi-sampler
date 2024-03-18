from typing import Tuple, Type

import numpy as np
import torch
from torch import nn

from vaepi_sampler.actions.action_types import LocalAction


class LossFunc(nn.Module):
    """General class that contains the structure of the ELBO loss function for the sampling considered in this repo"""

    def __init__(self, action: LocalAction):

        self.action = action

    def reconstruction_loss(
        self, z_in: torch.Tensor, z_out: torch.Tensor
    ) -> torch.Tensor:

        reconstruction = (1 / 2) * torch.mean(torch.sum((z_in - z_out) ** 2, dim=-1))

        return reconstruction

    def action_loss(self, x: torch.Tensor) -> torch.Tensor:

        action_part = torch.mean(self.action(x))
        norm_part = (1 / 2) * (self.action.N_T - 1) * np.log(2 * np.pi * self.action.dt)

        return norm_part + action_part

    def offset(self, z_in: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Part of the loss that does not depend on the variational params. We will include in the loss because then the loss becomes proportional to the free energy"""
        pass

    def trial_distribution_loss(
        self, params: Tuple, z_in: torch.Tensor, z_out: torch.Tensor, x: torch.Tensor
    ) -> torch.Tensor:

        pass

    def forward(
        self, params: Tuple, z_in: torch.Tensor, z_out: torch.Tensor, x: torch.Tensor
    ) -> torch.Tensor:
        """Computes the loss as a mean over the batch.
        args:
            params (Tuple of torch Tensors): parameters of a particular trial distribution of paths
            z_in (torch.Tensor): latent sample, should have shape (batch_size,latent_dim)
            z_out (torch.Tensor): reconstructed latent vector, same dim as z_in
            x: (torch.Tensor): path samples for the batch of shape (batch_size,N_T-2)
            action (instance of an Action class) action functional to use for the loss
        """
        action_loss = self.action_loss(x)
        reconstruction_loss = self.reconstruction_loss(z_in, z_out)
        trial_distribution_loss = self.trial_distribution_loss(params, z_in, z_out, x)
        offset = self.offset(z_in, x)

        loss = action_loss + reconstruction_loss + trial_distribution_loss + offset
        loss_dict = {
            "action_loss": action_loss,
            "reconstruction_loss": reconstruction_loss,
            "trial_distribution_loss": trial_distribution_loss,
            "offset": offset,
        }

        return loss, loss_dict


class GaussianElbo(LossFunc):
    """Loss for a Gaussian trial distribution at the output of the encoder"""

    def __init__(self, action: LocalAction):
        super().__init__(action)

    def trial_distribution_loss(
        self, params: Tuple, z_in: torch.Tensor, z_out: torch.Tensor, x: torch.Tensor
    ):

        mu_x = params[0]
        logvar_x = params[1]

        normalization_loss = (
            -1 / 2 * torch.mean(torch.sum(np.log(2 * np.pi) + logvar_x, dim=-1))
        )
        exponent_loss = -1 / 2 * x.shape[1]

        trial_loss = normalization_loss + exponent_loss

        return trial_loss

    def offset(self, z_in, x):

        return -1 / 2 * z_in.shape[1]
