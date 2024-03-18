from typing import List, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Parameter


class VAE_Base_LSTM(nn.Module):
    """This class represents a reverse VAE as described in [G. Cantwell arXiv:2209.10423]
    where the input is a latent variable, the encoder samples a path, and the decoder translates the path back to a latent variable.
    This architecture allows one to train a VAE to learn distributions without any available data.
    General class: contains skeleton structure still to be filled in with details
    """

    def __init__(
        self,
        x0: float,
        xf: float,
        latent_dim: int,
        lstm_hidden_size: int,
        encoder_base: nn.Module,
        encoder_sample: nn.Module,
        decoder_base: nn.Module,
        decoder_means: nn.Module,
        device=None,
    ):
        """
        args:
            encoder_base (nn.Module): base structure of the (reverse) encoder preprocessing the latent input either through some linear layers or other transformations
            encoder_sample (nn.Module): tail end of the encoder where the distribution variables are produced and the path is sampled with the reparametrization trick
            decoder_base (nn.Module): base structure of the (reverse) decoder processing the sampled path
            decoder_means (nn.Module): tail end of the decoder where the latent variable is reconstructed (means are taken as the reconstruction)
        """

        super(VAE_Base_LSTM, self).__init__()
        self.x0 = x0
        self.xf = xf
        self.encoder_base = encoder_base
        self.encoder_sample = encoder_sample
        self.decoder_base = decoder_base
        self.decoder_means = decoder_means
        self.latent_dim = latent_dim
        self.lstm_hidden_size = lstm_hidden_size

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def encode(
        self, z: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Encodes the latent variable and preprocesses it through a first base layer, and then passes it on to an autoregressive LSTM to sample the path.
        args:
            z (torch.Tensor) is the input tensor of the latent variable
        output:
            params: (One or Tuple of torch Tensors) parameters of the model path distribution
            path: (torch.Tensor) path sampled from the model distribution
        """

        zero_state = torch.zeros((z.shape[0], self.lstm_hidden_size)).to(self.device)
        start_first_cell = torch.full((z.shape[0], 1), self.x0).to(self.device)

        x = self.encoder_base(z)
        params, path = self.encoder_sample(
            input=start_first_cell,
            preprocessed_latent_input=x,
            state=(zero_state, zero_state),
        )

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
        self,
        z: torch.Tensor,
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor]:
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


class LSTMCell(nn.Module):

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = Parameter(torch.randn(4 * hidden_size, input_size))
        self.weight_hh = Parameter(torch.randn(4 * hidden_size, hidden_size))
        self.bias_ih = Parameter(torch.randn(4 * hidden_size))
        self.bias_hh = Parameter(torch.randn(4 * hidden_size))

    def forward(
        self, input: Tensor, state: Tuple[Tensor, Tensor]
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        hx, cx = state

        gates = (
            torch.mm(input, self.weight_ih.t())
            + self.bias_ih
            + torch.mm(hx, self.weight_hh.t())
            + self.bias_hh
        )
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * torch.tanh(cy)

        return hy, (hy, cy)


class LSTM_Encoder(nn.Module):

    def __init__(self, input_size, hidden_size, sequence_length: int):
        super().__init__()

        # here we add 1 to the input size because we are going to be concatenating two input arrays
        self.cell = LSTMCell(input_size + 1, hidden_size)
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length

        self.scale_param = Parameter(
            torch.ones(self.sequence_length, 2), requires_grad=True
        )
        self.final_layer = nn.Sequential(nn.Linear(hidden_size, 2))

    def forward(
        self,
        input: Tensor,
        preprocessed_latent_input: Tensor,
        state: Tuple[Tensor, Tensor],
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:

        outputs = []
        paths = []
        mu_x = []
        logvar_x = []

        input_cell = input

        for i in range(self.sequence_length):

            concat_input = torch.cat(
                (
                    input_cell,
                    torch.reshape(preprocessed_latent_input[:, i], (input.shape[0], 1)),
                ),
                dim=1,
            )

            # print(concat_input)

            out, state = self.cell(concat_input, state)
            outputs += [out]
            x = self.final_layer(out)

            mean = x[:, 0]
            logvar = x[:, 1]

            std_dev = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std_dev)
            sampled_values = mean + eps * std_dev

            paths += [sampled_values]
            mu_x += [mean]
            logvar_x += [logvar]

            sampled_values = torch.reshape(sampled_values, (input.shape[0], 1))
            input_cell = sampled_values

        outputs = torch.stack(outputs, dim=1)
        paths = torch.stack(paths, dim=1)
        mu_x = torch.stack(mu_x, dim=1)
        logvar_x = torch.stack(logvar_x, dim=1)

        return (mu_x, logvar_x), paths


class VAE_autoregressive_LSTM(VAE_Base_LSTM):

    def __init__(
        self,
        x0: float = 0.5,
        xf: float = 0.5,
        latent_dim: int = 1,
        hidden_size: int = 100,
        lstm_hidden_size: int = 16,
        encoder_output_size: int = 100,
        dropout: float = 0.1,
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
            nn.Linear(latent_dim, encoder_output_size),
            nn.ReLU(),
        )

        encoder_sample = LSTM_Encoder(
            input_size=1,
            hidden_size=lstm_hidden_size,
            sequence_length=encoder_output_size,
        )

        decoder_base = nn.Sequential(
            nn.BatchNorm1d(encoder_output_size),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout),
        )

        decoder_means = nn.Sequential(
            nn.BatchNorm1d(encoder_output_size),
            nn.Linear(hidden_size, latent_dim),
        )

        super(VAE_autoregressive_LSTM, self).__init__(
            x0=x0,
            xf=xf,
            latent_dim=latent_dim,
            lstm_hidden_size=lstm_hidden_size,
            encoder_base=encoder_base,
            encoder_sample=encoder_sample,
            decoder_base=decoder_base,
            decoder_means=decoder_means,
        )
