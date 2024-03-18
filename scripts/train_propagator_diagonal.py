import argparse
import logging
from utils import set_logger

logger = set_logger("logger")

import os

import mlflow
import numpy as np
import torch
from propagator_train_tools import train_propagator

from vaepi_sampler.actions.common_actions import HarmonicAction
from vaepi_sampler.models.autoregressive_lstm import VAE_autoregressive_LSTM
from vaepi_sampler.train.loss_functions import GaussianElbo
from vaepi_sampler.train.tools import sample_z_input

# CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device {device}")
if device.type == "cuda":
    logging.info(torch.cuda.get_device_name(0))


mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("propagator_diagonal")


hparams = {
    "latent_dim": 2,
    "num_epochs": 60,
    "lr": 1e-02,
    "batch_size": 64,
    "batches_per_epoch": 256,
    "dropout": 0.0,
    "weight_decay": 0.01,
    "lr_gamma_decay": 0.95,
    "lstm_hidden_size": 16,
}


T_MAX = 2
N_T = 100

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Script in which one or more propagators are trained"
    )
    parser.add_argument(
        "--T_MAX",
        default=T_MAX,
        type=float,
        help=f"Upper bound of the integral in the action (beta)",
    )
    parser.add_argument(
        "--N_T",
        default=N_T,
        type=float,
        help=f"Number of discretization points in the path",
    )
    args = parser.parse_args()

    # x_sample_space=np.linspace(-2,2,11,endpoint=True)
    x_sample_space = np.linspace(0.4, 2, 5, endpoint=True)

    for x in x_sample_space:

        sysparams = {"T_MAX": 2, "N_T": 100, "X0": x, "XF": x}

        train_propagator(
            hparams,
            sysparams,
            action_class=HarmonicAction,
            model_class=VAE_autoregressive_LSTM,
        )
