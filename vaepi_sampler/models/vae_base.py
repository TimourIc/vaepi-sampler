from typing import Tuple, Type, Union

import torch
import torch.nn as nn


class VAE_Base(nn.Module):

    """This class represents a reverse VAE as described in [G. Cantwell arXiv:2209.10423]
    where the input is a latent variable, the encoder samples a path, and the decoder translates the path back to a latent variable.
    This architecture allows one to train a VAE to learn distributions without any available data.
    General class: contains skeleton structure still to be filled in with details
    """

    def __init__(
        self,
        encoder_base: nn.Module,
        encoder_sample: nn.Module,
        decoder_base: nn.Module,
        decoder_means: nn.Module,
        latent_dim: int
    ):
        """
        args:
            encoder_base (nn.Module): base structure of the (reverse) encoder preprocessing the latent input either through some linear layers or other transformations
            encoder_sample (nn.Module): tail end of the encoder where the distribution variables are produced and the path is sampled with the reparametrization trick
            decoder_base (nn.Module): base structure of the (reverse) decoder processing the sampled path
            decoder_means (nn.Module): tail end of the decoder where the latent variable is reconstructed (means are taken as the reconstruction)
        """

        super(VAE_Base, self).__init__()
        self.encoder_base = encoder_base
        self.encoder_sample = encoder_sample
        self.decoder_base = decoder_base
        self.decoder_means = decoder_means
        self.latent_dim= latent_dim
        

    def encode(
        self,
        z: torch.Tensor,
    ) -> Tuple:
        """Encodes the latent variable and produces the parameters of the path distribution
        args:
            z (torch.Tensor) is the input tensor of the latent variable
        output:
            params: (One or Tuple of torch Tensors) parameters of the model path distribution
            path: (torch.Tensor) path sampled from the model distribution
        """

        x = self.encoder_base(z)
        params, path = self.encoder_sample(x)

        return params, path

    def decode(self, path: torch.Tensor) -> torch.Tensor:
        """Decodes the path back to the means of a Gaussian product distrubtion (mu_z is in practice taken as z_out themselves)
        args:
            x (torch.Tensor): path tensor after sampling it with self.sample_path()
        """

        x = self.decoder_base(path)
        mu_z = self.decoder_means(x)

        return mu_z

    def forward(
        self, z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full run through the reverse VAE, starting from a latent input it is reconstructed
        args:
            z (torch.Tensor) is the input tensor of the latent variable
        output:
            params: (one or Tuple of torch.Tensors) params produced for the model path distribution
            path: path sampled from the model path distribution
            z_out: reconstructed latent variable
        """

        params, path = self.encode(z)
        z_out = self.decode(path)

        return params, path, z_out
