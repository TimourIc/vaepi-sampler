import argparse
import torch.optim as optim
import yaml
from utils import set_logger, number_of_model_params
from datetime import datetime
import logging
logger = set_logger("logger")

import torch
from vaepi_sampler.actions.common_actions import HarmonicAction
from vaepi_sampler.models.gaussian_products import VAE_LSTM, VAE_FNN
from vaepi_sampler.train.loss_functions import GaussianElbo
from vaepi_sampler.train.tools import sample_z_input
 

from propagator_train_tools import train_propagator

import os

import mlflow

import numpy as np

#CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device {device}")
if device.type == "cuda":
    logging.info(torch.cuda.get_device_name(0))


mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("free_energy")


hparams={
        "latent_dim": 2,
        "num_epochs": 200,
        "lr" : 1e-02,
        "batch_size" : 128,
        "batches_per_epoch":256,
        "dropout": 0,
        "weight_decay":0.01,
        "lr_gamma_decay": 0.95,
        "upper_bound_logvar": 3,
        "lower_bound_logvar": -7
        }


T_MAX=2
N_T=100

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Script in which one or more propagators are trained")
    parser.add_argument("--T_MAX", default=T_MAX, type=float, help=f"Upper bound of the integral in the action (beta)")
    parser.add_argument("--N_T", default=N_T, type=float, help=f"Number of discretization points in the path")
    args = parser.parse_args()

    x_sample_space=np.linspace(-2,2,40,endpoint=True)

    for x in x_sample_space:

        sysparams={
            "T_MAX":args.T_MAX,
            "N_T":args.N_T,
            "X0": x,
            "XF": x
        }
        
    
        train_propagator(
            hparams,
            sysparams,
            action_class=HarmonicAction,
            model_class=VAE_FNN
        )
