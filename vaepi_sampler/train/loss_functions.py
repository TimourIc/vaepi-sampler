from typing import Type, Tuple
import numpy as np
import torch
from torch import nn
from vaepi_sampler.actions.action_types import LocalAction


class LossFunc(nn.Module):

    """General class that contains the structure of the ELBO loss function for the sampling considered in this repo"""

    def __init__(self, action: LocalAction):

        self.action=action

    def reconstruction_loss(self, z_in: torch.Tensor, z_out: torch.Tensor)-> torch.Tensor:

        reconstruction=(1/2)*torch.mean(torch.sum((z_in-z_out)**2, dim=1)) 

        return reconstruction
    
    def action_loss(self, x: torch.Tensor) -> torch.Tensor:

        return torch.mean(self.action(x))
    
    def trial_distribution_loss(self, params: Tuple, z_in: torch.Tensor, z_out: torch.Tensor, x: torch.Tensor)-> torch.Tensor:

        pass

    def forward(self,params: Tuple, z_in: torch.Tensor, z_out: torch.Tensor, x: torch.Tensor) -> torch.Tensor:

        """Computes the loss as a mean over the batch.
        args:
            params (Tuple of torch Tensors): parameters of a particular trial distribution of paths
            z_in (torch.Tensor): latent sample, should have shape (batch_size,latent_dim)
            z_out (torch.Tensor): reconstructed latent vector, same dim as z_in
            x: (torch.Tensor): path samples for the batch of shape (batch_size,N_T-2)
            action (instance of an Action class) action functional to use for the loss
        """

        loss=self.trial_distribution_loss(params,z_in,z_out,x)+self.action_loss(x)+self.reconstruction_loss(z_in,z_out)

        return loss 
    
class GaussianElbo(LossFunc):

    """Loss for a Gaussian trial distribution at the output of the encoder"""

    def __init__(self, action: LocalAction):
        super().__init__(action)
    
    def trial_distribution_loss(self, params:Tuple, z_in: torch.Tensor, z_out: torch.Tensor, x: torch.Tensor):
        
        mu_x=params[0]
        logvar_x=params[1]

        normalization_loss=-1/2*torch.mean(torch.sum(logvar_x ,dim=1))
        exponent_loss=-1/2*torch.mean(torch.sum(torch.div((x -mu_x )**2,torch.exp(logvar_x)) ,dim=1))

        trial_loss=normalization_loss+exponent_loss

        return trial_loss
 
# def gaussian_loss_func(z_in: torch.Tensor,  z_out: torch.Tensor, mu_x: torch.Tensor, logvar_x: torch.Tensor, x: torch.Tensor,  action: LocalAction) -> Tuple:


#     """Represents the loss function for the Gaussian path parametrization. If x0 and xf are fixed, should be an upper bound to the negative log of the Euclidean propagator.
#     Computes the loss as a mean over the batch.
#     args:
#         z_in (torch.Tensor): latent sample, should have shape (batch_size,latent_dim)
#         z_out (torch.Tensor): reconstructed latent vector, same dim as z_in
#         mu_x (torch.Tensor): means of the Gaussian parametrization for the path if end-points are fixed should have shape (batch_size,N_T-2)
#         logvar_x (torch.Tensor): logvars of the Gaussian parametrization
#         x: (torch.Tensor): path samples for the batch of shape (batch_size,N_T-2)
#         action (instance of an Action class) action functional to use for the loss
#     """

#     N=mu_x.size(-1)
#     offset=-N/2*np.log(2*np.pi)
#     action_loss=torch.mean(action(x))
#     reconstruction=(1/2)*torch.mean(torch.sum((z_in-z_out)**2, dim=1)) 
#     variance_loss=-1/2*torch.mean(torch.sum(logvar_x ,dim=1))
#     path_loss=-1/2*torch.mean(torch.sum(torch.div((x -mu_x )**2,torch.exp(logvar_x)) ,dim=1))
#     z_mag=-1/2*torch.mean(torch.sum(z_in**2,dim=1))

    
#     loss=variance_loss+path_loss+action_loss+reconstruction
#     free_energy= variance_loss+path_loss+action_loss +offset+z_mag

#     return loss, free_energy, path_loss, action_loss, reconstruction
