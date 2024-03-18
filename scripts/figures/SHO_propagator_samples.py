import argparse
from typing import Type

import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
import numpy as np

from vaepi_sampler.models.autoregressive_lstm import VAE_Base_LSTM
from vaepi_sampler.train.tools import minimal_path_SHO, sample_z_input

plt.rcParams["text.usetex"] = True


"""To run this script you need to have trained at least 4 propagators"""


def propagator_path_samples(
    model: Type[VAE_Base_LSTM],
    latent_dim: int,
    x0: float,
    xf: float,
    N_T: int,
    T_max: float,
    sample_size: int = 10,
):

    z_in = sample_z_input(sample_size, latent_dim)
    params, x, z_out = model(z_in)
    x_classical = minimal_path_SHO(N_T, T_max, x0=x0, xf=xf)
    x_np = x.detach().numpy()

    return x_np, x_classical


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Script in which a figure is shown of a trained propagator sampler"
    )
    parser.add_argument(
        "--experiment",
        default="propagators",
        type=str,
        help=f"experiment name in mlflow",
    )
    parser.add_argument("--run_id", type=str, help=f"mlflow id of the run")
    parser.add_argument(
        "--samples", default=100, type=int, help=f"number of samples to plot"
    )
    parser.add_argument("--fig_name", default="SHO_propagator_samples.png", type=str)
    args = parser.parse_args()

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("propagators")

    run_id = args.run_id

    # Load the model
    model = mlflow.pytorch.load_model(f"runs:/{run_id}/models")
    model.eval()
    # Get the value of the specified parameter
    x0 = float(mlflow.get_run(run_id).data.params.get("X0"))
    xf = float(mlflow.get_run(run_id).data.params.get("XF"))
    latent_dim = int(mlflow.get_run(run_id).data.params.get("latent_dim"))
    N_T = int(mlflow.get_run(run_id).data.params.get("N_T"))
    T_max = float(mlflow.get_run(run_id).data.params.get("T_MAX"))

    x_np, x_classical = propagator_path_samples(
        model, latent_dim, x0, xf, N_T, T_max, 50000
    )
    tarray = np.linspace(0, T_max, N_T, endpoint=True)

    fig, axes = plt.subplots(2, 1)
    ax1 = axes[0]
    ax2 = axes[1]

    for i in range(args.samples):
        sampled_path = np.concatenate((np.array([x0]), x_np[i, :], np.array([xf])))
        if i == 0:
            ax1.plot(tarray, sampled_path, label="sampler", linewidth=1)
        else:
            ax1.plot(tarray, sampled_path, linewidth=1)

    ax1.legend(loc="upper left")
    ax2.plot(tarray, x_classical, linewidth=3, color="orange", label="classical path")
    ax2.plot(
        tarray,
        np.concatenate((np.array([x0]), np.mean(x_np, axis=0), np.array([xf]))),
        label="average sampler path",
    )
    ax1.set_xlabel(r"$\tau/ \beta $", fontsize=12)
    ax1.set_ylabel(r"$x/ a_{ho}$", fontsize=14)
    ax2.set_xlabel(r"$\tau/ \beta $", fontsize=12)
    ax2.set_ylabel(r"$x/ a_{ho}$", fontsize=14)
    ax2.legend(loc="upper left")

    fig.tight_layout()
    fig.savefig(f"results/figures/{args.fig_name}", dpi=300)
