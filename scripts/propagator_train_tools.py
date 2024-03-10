import argparse
import torch.optim as optim
import yaml
from utils import set_logger, number_of_model_params
from datetime import datetime
import logging
 

import torch
from vaepi_sampler.actions.common_actions import LocalAction, HarmonicAction
from vaepi_sampler.models.gaussian_products import VAE_LSTM, VAE_FNN
from vaepi_sampler.models.vae_base import VAE_Base
from vaepi_sampler.train.loss_functions import GaussianElbo
from vaepi_sampler.train.tools import sample_z_input
from typing import Callable, Tuple, Type

 
import os

import mlflow

#CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device {device}")
if device.type == "cuda":
    logging.info(torch.cuda.get_device_name(0))
 


def mlflow_metric_logger(avg_train_loss: float, mu_x: torch.Tensor, var_x:torch.Tensor, epoch: int ):

    """Function to be called during training loop"""

    mlflow.log_metric("train_loss_epoch_avg",avg_train_loss, epoch)
    mlflow.log_metric("mu_x_zero",torch.mean(mu_x,dim=0)[0].item(), epoch)        
    mlflow.log_metric("mu_x_quarter",torch.mean(mu_x,dim=0)[int(mu_x.shape[1]/4)].item(), epoch)   
    mlflow.log_metric("mu_x_3quarter",torch.mean(mu_x,dim=0)[int(3*mu_x.shape[1]/4)].item(), epoch) 
    mlflow.log_metric("mu_x_final",torch.mean(mu_x,dim=0)[-1].item(), epoch)   
    mlflow.log_metric("var_x_zero",torch.mean(var_x,dim=0)[0].item(), epoch)       
    mlflow.log_metric("var_x_quarter",torch.mean(var_x,dim=0)[int(var_x.shape[1]/4)].item(), epoch)   
    mlflow.log_metric("var_x_3quarter",torch.mean(var_x,dim=0)[int(3*var_x.shape[1]/4)].item(), epoch) 
    mlflow.log_metric("var_x_final",torch.mean(var_x,dim=0)[-1].item(), epoch)
    mlflow.log_metric("avg_mu",torch.mean(mu_x).item() , epoch)
    mlflow.log_metric("avg_var",torch.mean(var_x).item(), epoch)

    pass


def train(
        num_epochs,
        batches_per_epoch,
        batch_size,
        optimizer,
        scheduler,
        loss_func,
        model,
        mlflow_metric_logger:Callable = None
    ):

    """Pytorch training run of the reverse VAE sampler, with the additional optional callable to log mlflow metrics."""
    
    logging.info(f"Starting training run for {model.__class__.__name__} wtih {number_of_model_params(model)} params")
    model.train()

    loss_vec=[]
    avg_loss_vec=[]
    param_vec=[]


    for epoch in range(num_epochs):

        avg_train_loss=0

        for batch_idx in range(batches_per_epoch):

            z_in=sample_z_input(batch_size,model.latent_dim)
            params, x, z_out  = model.forward(z_in)
            loss= loss_func.forward(params, z_in, z_out, x)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            avg_train_loss=avg_train_loss*(batch_idx)/(batch_idx+1)+loss.data.item()/(batch_idx+1)
    
        scheduler.step()
        
        mu_x=params[0].data
        var_x=torch.exp(params[1]).data

        loss_vec.append(loss)
        avg_loss_vec.append(avg_train_loss)
        param_vec.append(params)

        print(100*"*")
        logging.info(f"finished epoch:{epoch}/{num_epochs}")
        logging.info(f"average train loss: {avg_train_loss}")
        logging.info(f"avg mu_x (along path) last batch: {torch.mean(params[0])}")
        logging.info(f"avg var_x (along path) last batch: {torch.mean(torch.exp(params[1]))}")

        if mlflow_metric_logger is not None:
            mlflow_metric_logger(avg_train_loss, mu_x, var_x, epoch)

    return loss_vec, avg_loss_vec, param_vec


def train_propagator(hparams:dict, 
                     sysparams: dict,
                     action_class: Type[LocalAction],
                     model_class: Type[VAE_Base]
                     ):
    
    """Sets up and calls a training run of the propagator with given params"""

    #setup training classes
    action=action_class(T_max=sysparams["T_MAX"],N_T=sysparams["N_T"], x0=sysparams["X0"], xf=sysparams["XF"])
    model=model_class(latent_dim=hparams["latent_dim"], hidden_size=sysparams["N_T"]-2, encoder_output_size=sysparams["N_T"]-2, dropout=hparams["dropout"] , upper_bound_logvar=hparams["upper_bound_logvar"], lower_bound_logvar=hparams["lower_bound_logvar"])
    OPTIMIZER=optim.Adam(model.parameters(), lr=hparams["lr"], weight_decay=hparams["weight_decay"])
    SCHEDULER = optim.lr_scheduler.ExponentialLR(OPTIMIZER, gamma=hparams["lr_gamma_decay"])
    LOSS_FUNC= GaussianElbo(action=action)

    #add the choice of loss funcs/optimizers to the dict to be tracked in mlflow
    hparams.update({"optimizer" : OPTIMIZER.__class__.__name__,
        "scheduler": SCHEDULER.__class__.__name__,
        "loss_func": LOSS_FUNC.__class__.__name__})

    #training run with metric logging to mlflow
    with mlflow.start_run(run_name="VAE_FNN"):
        loss_vec, avg_loss_vec, param_vec=train(
                num_epochs= hparams["num_epochs"],
                batches_per_epoch=hparams["batches_per_epoch"],
                batch_size=hparams["batch_size"],
                optimizer=OPTIMIZER,
                scheduler=SCHEDULER,
                loss_func=LOSS_FUNC,
                model=model,
                mlflow_metric_logger=mlflow_metric_logger
                )

    
        mlflow.log_params(params=hparams)
        mlflow.log_params(params=sysparams)

        #save progression of params throughout the training run as an artifact
        torch.save(param_vec,f"results/temp/param_tensor.pt")
        mlflow.log_artifact(f"results/temp/param_tensor.pt")
        os.remove(f"results/temp/param_tensor.pt")

        #log model
        mlflow.pytorch.log_model(model,"models")

 

 



 


 
 
