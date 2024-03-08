import torch
import numpy as np
import logging
from vaepi_sampler.models.vae_base import VAE_Base
from vaepi_sampler.train.loss_functions import LossFunc
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

def sample_z_input(batch_size,latent_dim) -> torch.Tensor:

    "samples the latent variable at the input of the (reverse) VAE"

    return torch.randn((batch_size,latent_dim)) 


def minimal_path_SHO(N_T, T_max, x_0):

    "for equal starting points x0=xf, returns the minimal Euclidean path for an SHO"

    x=[x_0]
    dt=T_max/(N_T-1)

    for i in range(1,N_T):
        x.append(x_0*np.cosh(i*dt)+x_0*(1-np.cosh(T_max))/np.sinh(T_max)*np.sinh(i*dt))

    return x



 
    