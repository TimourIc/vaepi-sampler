import matplotlib.pyplot as plt
from vaepi_sampler.train.tools import sample_z_input, minimal_path_SHO
from vaepi_sampler.models.vae_base import VAE_Base
from typing import Type
import numpy as np
import mlflow
import mlflow.pytorch

plt.rcParams['text.usetex'] = True


"""To run this script you need to have trained at least 4 propagators"""


def propagator_path_samples(model: Type[VAE_Base], latent_dim: int, x0: float , xf: float, N_T: int, T_max: float, sample_size: int=10):

    z_in=sample_z_input(sample_size,latent_dim)
    params, x, z_out= model(z_in)
    x_classical=minimal_path_SHO(N_T, T_max, x0=x0, xf=xf)
    x_np = x.detach().numpy()

    return x_np, x_classical


if __name__=="__main__":

    fig, axes=plt.subplots(2,2)

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("propagators")
    runs=mlflow.search_runs( )

    for index, run in runs.head(4).iterrows():
        ax=axes.ravel()[index]
        run_id = run['run_id']

        # Load the model
        with mlflow.start_run(run_id=run_id):
            model = mlflow.pytorch.load_model(f"runs:/{run_id}/models")

            # Get the value of the specified parameter
            x0 = float(mlflow.get_run(run_id).data.params.get("X0"))
            xf=  float(mlflow.get_run(run_id).data.params.get("XF"))
            latent_dim=  int(mlflow.get_run(run_id).data.params.get("latent_dim"))
            N_T= int(mlflow.get_run(run_id).data.params.get("N_T"))
            T_max=float(mlflow.get_run(run_id).data.params.get("T_MAX"))
                
        
        x_np, x_classical=propagator_path_samples(model, latent_dim, x0, xf, N_T, T_max, 20)
        tarray=np.linspace(0,T_max,N_T, endpoint=True)
        
        for i in range(x_np.shape[0]):
            if i==0:
                ax.plot(tarray,np.concatenate((np.array([x0]), x_np[i,:],np.array([xf]))), label="sampler")
            else:
                ax.plot(tarray,np.concatenate((np.array([x0]), x_np[i,:],np.array([xf]))))
        ax.plot(tarray,x_classical, linewidth=3, color="orange", label="classical path")
        ax.set_xlabel(r"$\tau/ \beta $", fontsize=12)
        ax.set_ylabel(r"$x/ a_{ho}$", fontsize=14)
        if index==1:
            ax.legend(loc="upper left")

    fig.tight_layout()
    fig.savefig(f"results/figures/SHO_propagator_samples.png", dpi=300)
            
    


 

 

