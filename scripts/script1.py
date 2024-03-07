import argparse
from typing import Type
import yaml
from utils import set_logger
import numpy as np
from vaepi_sampler.actions.common_actions import HarmonicAction
from vaepi_sampler.models.vae_base import VAE_base, VAE_LSTM 
import torch
from vaepi_sampler.actions.action_types import LocalAction
import torch.optim as optim
import matplotlib.pyplot as plt

# config_params:
with open("config/config.yaml", "r") as file:
    config_data = yaml.safe_load(file)
data_path = config_data["paths"]["data_path"]
results_path = config_data["paths"]["results_path"]
logger = set_logger("logger")


def loss_func(z_in: torch.Tensor,  mu_x: torch.Tensor, logvar_x: torch.Tensor, x: torch.Tensor, z_out: torch.Tensor , action: Type[LocalAction]):

    W=10**6

    """Computes the average loss or beta*free energy of an input batch"""

    N_T=mu_x.size(-1)
    offset=-N_T/2*np.log(2*np.pi)
    action_loss=torch.mean(action(x))
    reconstruction=(1/2)*torch.mean(torch.sum((z_in-z_out)**2, dim=1)) 
    z_in_magnitude=(1/2)*torch.mean(torch.sum((z_in)**2, dim=1)) 
    variance_loss=-1/2*torch.mean(torch.sum(logvar_x[:,1:-1],dim=1))
    path_loss=-1/2*torch.mean(torch.sum((x[:,1:-1]-mu_x[:,1:-1])**2/torch.exp(logvar_x[:,1:-1]),dim=1))

     
    loss=reconstruction+variance_loss+path_loss+action_loss 

 
    

    return loss,  path_loss, variance_loss, z_in_magnitude, reconstruction, action_loss

def sample_z_input(batch_size,latent_dim) -> torch.Tensor:

    "samples the latent variable at the input of the (reverse) VAE"

    return torch.randn((batch_size,latent_dim)) 


def loss_func_test(z_in: torch.Tensor,  mu_x: torch.Tensor, logvar_x: torch.Tensor, x: torch.Tensor,  action: Type[LocalAction]):

    W=10**6

    """Computes the average loss or beta*free energy of an input batch"""

    N_T=mu_x.size(-1)
    offset=-N_T/2*np.log(2*np.pi)
    action_loss=torch.mean(action(x))
    variance_loss=-1/2*torch.mean(torch.sum(logvar_x[:,1:-1],dim=1))
    path_loss=-1/2*torch.mean(torch.sum((x[:,1:-1]-mu_x[:,1:-1])**2/torch.exp(logvar_x[:,1:-1]),dim=1))

 
    loss=variance_loss+path_loss+action_loss 

    return loss,   path_loss, variance_loss, action_loss


def minimal_path(N_T,dt , T_max, x_0):

    x=[x_0]

    for i in range(1,N_T):
        x.append(x_0*np.cosh(i*dt)+x_0*(1-np.cosh(T_max))/np.sinh(T_max)*np.sinh(i*dt))
 
 
    return x

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="description of script")
    parser.add_argument("--ARGUMENT", default="Hello World", type=str, help=f"ARGUMENT")
    args = parser.parse_args()
    # print(args.ARGUMENT)




    
    T_max=4
    N_T=200
    num_batches=200000
    loss_vec=[]
    train_loss=0
    batch_size=8
    latent_dim=2
    x0=0.5
    action=HarmonicAction(T_max=T_max,N_T=100)
    vae_model=VAE_test(latent_dim=latent_dim, encoder_hidden_size=N_T, encoder_output_size=N_T,     dropout=0.2, x0=x0)
    optimizer = optim.Adam(vae_model.parameters(), lr=1e-5, weight_decay=0.001 )
    # optimizer = optim.SGD(vae_model.parameters(), lr=1e-10 , weight_decay=0.01)
    vae_model.train()

    for batch_idx in range(num_batches):
        z_in=sample_z_input(batch_size,latent_dim)
        mu_x, logvar_x, x  = vae_model.forward(z_in)
        # loss, closed_path_loss, path_loss, variance_loss, z_in_magnitude, reconstruction, action_loss=loss_func_test(z_in,mu_x,logvar_x,x, action)
        loss,   path_loss, variance_loss, action_loss= loss_func_test(z_in,mu_x,logvar_x,x, action)
        loss.backward()
        optimizer.step()
        if batch_idx%500==0:
            print(100*"*")
            print(f"iteration: {batch_idx}/{num_batches}")
            print(f"loss: {loss/T_max}")
 
            print(f"action loss {action_loss/T_max}")
            print(f"kinetic part: {torch.mean(action.kinetic_part(x))/T_max}")
            print(f"pot part: {torch.mean(action.potential_part(x))/T_max}")
            print(f"min logvar : {torch.min(logvar_x)}")
            print(f"mean var : {torch.mean(torch.exp(logvar_x))}")
            print(f"mu_x : {torch.mean(mu_x)}")
            # print(f"pot part: {torch.mean(action.potential_part(x))/T_max}")
            # print(f"logvar term: {variance_loss/T_max}")
            # print(f"path_loss term: {path_loss/T_max}")
            x_np = x.detach().numpy()
            mu_np=mu_x.detach().numpy()

        if batch_idx%10000==0:
            for i in range(x_np.shape[0]):
                plt.plot(x_np[i, :])
                plt.scatter(range(len(mu_np[i,:])),mu_np[i,:])
            x_min=minimal_path(N_T,T_max/(N_T-1) , T_max, x0)
            plt.plot(x_min, linewidth=3, color="orange")
            plt.show()

 


    
        # print(torch.sum(action.V(((x[:, :-1] + x[:, 1:]) / 2)),dim=1))
        

    
 
 




