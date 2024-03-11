from typing import Tuple, Union

import torch
import torch.nn as nn

from vaepi_sampler.models.vae_base import VAE_Base

"""This module containins functionalities and the VAE samplers themselves that parametrize the paths as a product of Gaussians.
This is of course a simplified first step and the goal is to go to more advanced parametrizations.

GaussianProductSampler: one part of the VAE that encodes the Gaussian params and samples the path. Should is to be passed to a VAE_Base class as encoder_sampler.
VAE_FNN: functional feedforward VAE sampler where everything is a dense layer
VAE_LSTM: functional VAE sampler where an LSTM is added in the encoder base. 
""" 


class GaussianProductEncoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        lower_bound_logvar: float = -3,
        upper_bound_logvar: float = 3,
    ):
        """Represents the tail of the encoder. Gven a preprocessed input x, creates the params of a Gaussian product distribution and samples it to produce a path.
        args:
            input_size (int): size of the preprocessed input
            output_size (int): size of the output -- number of points to be sampled in the path
            lower/upper bounds logvars (floats): useful to be able to constraint variance for stability
        """

        super(GaussianProductEncoder, self).__init__()

        self.lower_bound_logvar = lower_bound_logvar
        self.upper_bound_logvar = upper_bound_logvar

        self.means = nn.Sequential(
            nn.BatchNorm1d(input_size), nn.Linear(input_size, output_size)
        )

        self.logvars = nn.Sequential(
            nn.BatchNorm1d(input_size),
            nn.Linear(input_size, output_size),
            nn.Sigmoid(),
        )

    def create_params(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        """Yields the means and logvars tensors for the Gaussian product distribution.
        Dimensions of (B,output_size), with output_size being the number of points of the path that will be sampled.

        args:
            x (torch.Tensor): NOT the path, is a preprocessed state that already went through the encoder_base layer.
        output:
            params (Tuple of torch.Tensors) means and logvar tensors.
        """

        mu_x = self.means(x)
        logvar_x = self.logvars(x)
        logvar_x = (
            self.upper_bound_logvar - self.lower_bound_logvar
        ) * logvar_x + self.lower_bound_logvar
        params = (mu_x, logvar_x)

        return params

    def sample_path(self, params: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Uses parameters from the create_params method to sample a path from the Gaussian product distribution using the reparametrization trick.

        args:
            params (Tuple of torch.Tensor): a tuple of mu_x and logvar_x tensors.
        output:
            x (torch.Tensor): a sampled path from the distribution
        """

        mu_x = params[0]
        logvar_x = params[1]

        std_x = torch.exp(0.5 * logvar_x)
        eps = torch.randn_like(std_x)
        path = eps.mul(std_x).add_(mu_x)

        return path

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        params = self.create_params(x)
        path = self.sample_path(params)

        return params, path


class VAE_FNN(VAE_Base):

    

    def __init__(
        self,
        latent_dim: int = 4,
        hidden_size: int = 100,
        encoder_output_size: int = 100,
        dropout: float = 0.1,
        lower_bound_logvar: float = -3,
        upper_bound_logvar: float = 3,
    ):
        """Complete and functional VAE that uses simple dense layers everywhere.
        args:
                latent_dim (int): size of the latent input/output vectors
                hidden_size (int): size of the hidden layers in the encoder and decoder, all taken the same
                encoder_output_size (int): number of elements in the path discretization (will provide this number of means and variances for the distribution)
                dropout (0<float<1): dropout probability during training
                lower_bound_logvar (float): limit log-variance to some lower bound (good for stability)
                upper_bound_logvar (float): limit log-variance to some upper bound
        """

        encoder_base = nn.Sequential(
            nn.BatchNorm1d(latent_dim),
            nn.Linear(latent_dim, hidden_size),
            nn.ReLU(),
            # nn.BatchNorm1d(hidden_size),
            # nn.Linear(hidden_size, hidden_size),
            # nn.ReLU(),
        )

        encoder_sampler = GaussianProductEncoder(
            input_size=hidden_size,
            output_size=encoder_output_size,
            lower_bound_logvar=lower_bound_logvar,
            upper_bound_logvar=upper_bound_logvar,
        )

        decoder_base = nn.Sequential(
            nn.BatchNorm1d(encoder_output_size),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout),
        )

        decoder_means = nn.Sequential(
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, latent_dim),
        )

        super(VAE_FNN, self).__init__(
            encoder_base, encoder_sampler, decoder_base, decoder_means, latent_dim
        )


class VAE_LSTM(VAE_Base):

    def __init__(
        self,
        latent_dim: int = 4,
        hidden_size: int = 100,
        encoder_output_size: int = 100,
        dropout: float = 0.1,
        lower_bound_logvar: float = -3,
        upper_bound_logvar: float = 3,
        lstm_hidden_size:int = 4
    ):
        """Complete and functional VAE that uses an LSTM in its encoder to encourage the sequential generation of the path.
        args:
                latent_dim (int): size of the latent input/output vectors ,
                lstm_hidden_size (int): size of the hidden lstm vector
                encoder_hidden_size (int): size of the hidden layers in the encoder and decoder, all taken the same
                encoder_output_size (int): number of elements in the path discretization (will provide this number of means and variances for the distribution)
                dropout (0<float<1): dropout probability during training
                lower_bound_logvar (float): limit log-variance to some lower bound (good for stability)
                upper_bound_logvar (float): limit log-variance to some upper bound
        """

        encoder_base = nn.Sequential(
            nn.LSTM(
                input_size=latent_dim,
                hidden_size=lstm_hidden_size,
                num_layers=1,
                batch_first=True,
            )
        )

        encoder_sampler = GaussianProductEncoder(
            input_size=hidden_size * lstm_hidden_size,
            output_size=encoder_output_size,
            lower_bound_logvar=lower_bound_logvar,
            upper_bound_logvar=upper_bound_logvar,
        )

        decoder_base = nn.Sequential(
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout),
        )

        decoder_means = nn.Sequential(
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, latent_dim),
        )

        super(VAE_LSTM, self).__init__(
            encoder_base, encoder_sampler, decoder_base, decoder_means, latent_dim
        )

        self.encoder_output_size = encoder_output_size

    def encode(self, z: torch.Tensor) -> Tuple:

        """Overwrite the VAE_Base encode method because (1) the input z_in needs to be reshaped and (2) the LSTM produces a tuple and we want to pass only the first component"""

        z_in_extended = z.unsqueeze(1).expand(
            z.shape[0], self.encoder_output_size, z.shape[1]
        )
        x, _ = self.encoder_base(z_in_extended)
        x = x.reshape(x.size(0), -1)
        params, path = self.encoder_sample(x)

        return params, path
