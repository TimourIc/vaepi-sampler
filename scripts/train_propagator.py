import argparse
import logging

from utils import number_of_model_params, set_logger

logger = set_logger("logger")

import mlflow
import torch
from propagator_train_tools import train_propagator

from vaepi_sampler.actions.common_actions import HarmonicAction
from vaepi_sampler.models.autoregressive_lstm import VAE_autoregressive_LSTM

# CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device {device}")
if device.type == "cuda":
    logging.info(torch.cuda.get_device_name(0))


mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("propagators")


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

sysparams = {"T_MAX": 2, "N_T": 100, "X0": -1, "XF": 1}


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Script in which one or more propagators are trained"
    )
    parser.add_argument(
        "--T_MAX",
        default=sysparams["T_MAX"],
        type=float,
        help=f"Upper bound of the integral in the action (beta)",
    )
    parser.add_argument(
        "--N_T",
        default=sysparams["N_T"],
        type=int,
        help=f"Number of discretization points in the path",
    )
    parser.add_argument(
        "--X0",
        default=sysparams["X0"],
        type=float,
        help=f"Starting point of the propagator",
    )
    parser.add_argument(
        "--XF",
        default=sysparams["XF"],
        type=float,
        help=f"Final point of the propagator",
    )
    args = parser.parse_args()

    sysparams["T_MAX"] = args.T_MAX
    sysparams["N_T"] = args.N_T
    sysparams["X0"] = args.X0
    sysparams["XF"] = args.XF

    train_propagator(
        hparams,
        sysparams,
        action_class=HarmonicAction,
        model_class=VAE_autoregressive_LSTM,
    )
