import argparse
from typing import Type

import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
import numpy as np
import torch
from scipy.integrate import quad
from scipy.interpolate import CubicSpline

from vaepi_sampler.actions.common_actions import HarmonicAction
from vaepi_sampler.models.autoregressive_lstm import VAE_Base_LSTM
from vaepi_sampler.train.loss_functions import GaussianElbo
from vaepi_sampler.train.tools import minimal_path_SHO, sample_z_input

plt.rcParams["text.usetex"] = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""To run this script you need to have trained at least 4 propagators"""


def SHO_propagator(x0, xf, m=1, omega=1, T_max=1, hbar=1):

    propagator = np.sqrt(
        m * omega / (2 * np.pi * hbar * np.sinh(hbar * omega * T_max))
    ) * np.exp(
        -m
        * omega
        / (2 * hbar * np.sinh(hbar * omega * T_max))
        * ((x0**2 + xf**2) * np.cosh(hbar * omega * T_max) - 2 * x0 * xf)
    )

    return propagator


def get_free_energy(x, K, T_max, x_max):

    spline_interpolated = CubicSpline(x, K)
    integral, error = quad(spline_interpolated, -x_max, x_max)
    free_energy = -np.log(integral) / T_max

    return free_energy


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Script in which a figure is shown of a trained propagator sampler"
    )
    parser.add_argument(
        "--experiment",
        default="propagator_diagonal",
        type=str,
        help=f"experiment name in mlflow",
    )
    parser.add_argument(
        "--samples", default=10000, type=int, help=f"number of samples to plot"
    )
    parser.add_argument("--fig_name", default="SHO_propagator_diagonal.png", type=str)
    args = parser.parse_args()

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(args.experiment)
    runs = mlflow.search_runs()

    lossvec = []
    xvec = []

    for index, run in runs.iterrows():
        run_id = run["run_id"]
        # Load the model

        with mlflow.start_run(run_id=run_id):

            model = mlflow.pytorch.load_model(f"runs:/{run_id}/models")
            x0 = float(mlflow.get_run(run_id).data.params.get("X0"))
            xf = float(mlflow.get_run(run_id).data.params.get("XF"))
            latent_dim = int(mlflow.get_run(run_id).data.params.get("latent_dim"))
            N_T = int(mlflow.get_run(run_id).data.params.get("N_T"))
            T_max = float(mlflow.get_run(run_id).data.params.get("T_MAX"))
            z_in = sample_z_input(100000, latent_dim=latent_dim).to(device)
            model.eval()
            params, x, z_out = model(z_in)
            action = HarmonicAction(T_max=T_max, N_T=N_T, x0=x0, xf=xf)
            loss_func = GaussianElbo(action=action)
            loss, _ = loss_func.forward(params, z_in, z_out, x)
            xvec.append(x0)
            lossvec.append(loss.cpu().detach().numpy())

    xvec = np.array(xvec)
    lossvec = np.array(lossvec)

    fig = plt.figure()
    x_prop = np.linspace(-2, 2, 100)
    K_vec_real = SHO_propagator(x_prop, x_prop, T_max=T_max)
    sorted_indices = np.argsort(xvec)
    xvec = xvec[sorted_indices]
    K_vec = np.exp(-lossvec)[sorted_indices]
    plt.scatter(xvec, K_vec, label=r"sampler $e^{-\textit{loss}}$", color="red")
    plt.plot(x_prop, K_vec_real, label=r"analytic diag. $K(x|x)$")
    plt.legend(loc="upper right")
    plt.ylim([0, 0.25])
    plt.xlabel(r"$x/a_{ho}$")
    plt.ylabel(r"$K~a_{ho}$")

    fig.tight_layout()
    fig.savefig(f"results/figures/{args.fig_name}", dpi=300)

    F_sampler = get_free_energy(xvec, K_vec, T_max=T_max, x_max=2)
    F_real = get_free_energy(x_prop, K_vec_real, T_max=T_max, x_max=2)
    print(f"free energy sampler {F_sampler}")
    print(f"free energy analytic {F_real}")
