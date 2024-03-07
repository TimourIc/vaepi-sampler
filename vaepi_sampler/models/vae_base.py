import torch
from typing import Tuple
import torch.nn as nn
 
class VAE_base(nn.Module):

    """This class represents a reverse VAE as described in [G. Cantwell arXiv:2209.10423]
    where the input is a latent variable, the encoder samples a path, and the decoder translates the path back to a latent variable.
    This architecture allows one to train a VAE to learn distributions without any available data.
    Base model: simple feedforward encoders and decoders
    """

    def __init__(self,
                 latent_dim:int=4,
                 hidden_size:int=100,
                 encoder_output_size:int=100,
                 dropout:float=0.1,
                 lower_bound_logvar:float=-3,
                 upper_bound_logvar:float=3
                 ):
        
        """
        args:
                latent_dim (int): size of the latent input/output vectors 
                hidden_size (int): size of the hidden layers in the encoder and decoder, all taken the same
                encoder_output_size (int): number of elements in the path discretization (will provide this number of means and variances for the distribution)
                dropout (0<float<1): dropout probability during training
                lower_bound_logvar (float): limit log-variance to some lower bound (good for stability)
                upper_bound_logvar (float): limit log-variance to some upper bound
        """

        super(VAE_base, self).__init__()

        self.latent_dim=latent_dim
        self.hidden_size=hidden_size
        self.encoder_output_size=encoder_output_size
        self.upper_bound_logvar=upper_bound_logvar
        self.lower_bound_logvar=lower_bound_logvar

        
        self.encoder_base = nn.Sequential(
        nn.BatchNorm1d(latent_dim),
        nn.Linear(latent_dim, hidden_size),
        nn.ReLU(),
        nn.Dropout(p=dropout),
        nn.BatchNorm1d(hidden_size),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Dropout(p=dropout),
        )

        self.encoder_means=nn.Sequential(
        nn.BatchNorm1d(hidden_size),
        nn.Linear(hidden_size, encoder_output_size)
        )

        self.encoder_logvars=nn.Sequential(
        nn.BatchNorm1d(hidden_size),
        nn.Linear(hidden_size, encoder_output_size),
        nn.Sigmoid(),
        )

        self.decoder_base=nn.Sequential(
        nn.BatchNorm1d(hidden_size),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Dropout(p=dropout)
        )

        self.decoder_means=nn.Sequential(
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, latent_dim),
        )


    def encode(self,z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        """Encodes the latent variable and produces the parameters of the path distribution
        args: 
            z (torch.Tensor) is the input tensor of the latent variable 
        """

        x=self.encoder_base(z)
        mu_x=self.encoder_means(x)
        logvar_x=self.encoder_logvars(x) 

        logvar_x=(self.upper_bound_logvar-self.lower_bound_logvar)*logvar_x+self.lower_bound_logvar

        return mu_x, logvar_x
    
    
    def sample_path(self, mu_x: torch.Tensor, logvar_x: torch.Tensor) -> torch.Tensor:

        """samples the (Gaussian) path from the encoded parameters using the reparametrization trick
        args:
            mu_x (torch.Tensor): the means tensor coming from the encoder layer 
            logvar_x (torch.Tensor): the logvar tensor coming from the encoder layer
        """

        std_x=torch.exp(0.5*logvar_x)
        eps=torch.randn_like(std_x)
        path=eps.mul(std_x).add_(mu_x)

        return path
    
    def decode(self, x: torch.Tensor) -> torch.Tensor:

        """Decodes the path back to the means of a Gaussian product distrubtion (mu_z is in practice taken as z_out themselves)
        args:
            x (torch.Tensor): path tensor after sampling it with self.sample_path()
        """

        x=self.decoder_base(x)
        mu_z=self.decoder_means(x)

        return mu_z
    
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        """Full run through the reverse VAE, starting from a latent input it is reconstructed
        args: 
            z (torch.Tensor) is the input tensor of the latent variable 
        """
        
        mu_x, logvar_x = self.encode(z)
        x=self.sample_path(mu_x, logvar_x)
        z_out=self.decode(x)

        return mu_x, logvar_x, x, z_out
    


class VAE_LSTM(nn.Module):

    """This class represents a reverse VAE as described in [G. Cantwell arXiv:2209.10423]
    where the input is a latent variable, the encoder samples a path, and the decoder translates the path back to a latent variable.
    This architecture allows one to train a VAE to learn distributions without any available data
    Here, we also use ideas from [Y. Che et al., Phys. Rev. B 105, 214205] to use an LSTM in the architecture to produce the means and variances.
    """

    def __init__(self,
                 latent_dim:int=4,
                 lstm_hidden_size:int=8,
                 hidden_size:int=100,
                 encoder_output_size:int=100,
                 dropout:float=0.1,
                 lower_bound_logvar:float=-3,
                 upper_bound_logvar:float=3
                 ):
        
        """
        args:
                latent_dim (int): size of the latent input/output vectors 
                lstm_hidden_size(int): size of the hidden states in the LSTM
                hidden_size (int): size of the other hidden layers in the encoder and decoder, all taken the same
                encoder_output_size (int): number of elements in the path discretization (will provide this number of means and variances for the distribution)
                dropout (0<float<1): dropout probability during training
                lower_bound_logvar (float): limit log-variance to some lower bound (good for stability)
                upper_bound_logvar (float): limit log-variance to some upper bound
        """

        super(VAE_LSTM, self).__init__()

        self.latent_dim=latent_dim
        self.encoder_output_size=encoder_output_size
        self.upper_bound_logvar=upper_bound_logvar
        self.lower_bound_logvar=lower_bound_logvar
        self.lstm_hidden_size=lstm_hidden_size

        self.encoder_base = nn.Sequential(
        nn.LSTM(input_size=latent_dim, hidden_size=lstm_hidden_size, num_layers=1, batch_first=True),
        )

        self.encoder_means=nn.Sequential(
        nn.BatchNorm1d(hidden_size),
        nn.Dropout(p=dropout),
        nn.Linear(hidden_size, encoder_output_size)
        )

        self.encoder_logvars=nn.Sequential(
        nn.BatchNorm1d(hidden_size),
        nn.Dropout(p=dropout),
        nn.Linear(hidden_size, encoder_output_size),
        nn.Sigmoid(),
        )

        self.decoder_base=nn.Sequential(
        nn.BatchNorm1d(hidden_size),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Dropout(p=dropout),
        )

        self.decoder_means=nn.Sequential(
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, latent_dim),
        )


    def encode(self,z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        """Encodes the latent variable and produces the parameters of the path distribution
        args: 
            z (torch.Tensor) is the input tensor of the latent variable 
        """

        #extend the latent input such that it is used as an input in every LSTM cell
        z_in_extended=z.unsqueeze(1).expand(z.shape[0], self.encoder_output_size, z.shape[1]) 

        x, _=self.encoder_base(z_in_extended)

        mu_x=self.encoder_means(x[:,:,0])
        logvar_x=self.encoder_logvars(x[:,:,1]) 
        logvar_x=(self.upper_bound_logvar-self.lower_bound_logvar)*logvar_x+self.lower_bound_logvar

        return mu_x, logvar_x
    
    
    def sample_path(self, mu_x: torch.Tensor, logvar_x: torch.Tensor) -> torch.Tensor:


        """samples the (Gaussian) path from the encoded parameters using the reparametrization trick
        args:
            mu_x (torch.Tensor): the means tensor coming from the encoder layer 
            logvar_x (torch.Tensor): the logvar tensor coming from the encoder layer
        """

        std_x=torch.exp(0.5*logvar_x)
        eps=torch.randn_like(std_x)

        path=eps.mul(std_x).add_(mu_x)

        return path
    
    def decode(self, x: torch.Tensor) -> torch.Tensor:

        """Decodes the path back to the means of a Gaussian product distrubtion (mu_z is in practice taken as z_out themselves)
        args:
            x (torch.Tensor): path tensor after sampling it with self.sample_path()
        """

        x=self.decoder_base(x)
        mu_z=self.decoder_means(x)

        return mu_z
    
    def forward(self, z_in: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        """Full run through the reverse VAE, starting from a latent input it is reconstructed
        args: 
            z (torch.Tensor) is the input tensor of the latent variable 
        """
        
        mu_x, logvar_x = self.encode(z_in)
        x=self.sample_path(mu_x, logvar_x)
        z_out=self.decode(x)

        return mu_x, logvar_x, x, z_out
    
        

