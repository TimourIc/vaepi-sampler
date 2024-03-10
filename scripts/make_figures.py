import matplotlib.pyplot as plt
from vaepi_sampler.train.tools import sample_z_input, minimal_path_SHO
from vaepi_sampler.models.vae_base import VAE_Base
from typing import Type
import numpy as np
import mlflow
import mlflow.pytorch

def propagator_path_plot(model: Type[VAE_Base], latent_dim: int, x0: float , xf: float, N_T: int, T_max: float, sample_size: int=10):

    z_in=sample_z_input(sample_size,latent_dim)
    params, x, z_out= model(z_in)
    x_classical=minimal_path_SHO(N_T, T_max, x0=x0, xf=xf)

    
    x_np = x.detach().numpy()

    for i in range(x_np.shape[0]):
        plt.plot(np.concatenate((np.array([x0]), x_np[i,:],np.array([xf]))))
    plt.plot(x_classical, linewidth=3, color="orange")
    plt.show()

if __name__=="__main__":

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("propagators")
    runs=mlflow.search_runs( )
    for index, run in runs.iterrows():
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
                
        
        propagator_path_plot(model, latent_dim, x0, xf, N_T, T_max, 40)

 

