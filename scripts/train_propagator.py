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

from torch.utils.tensorboard import SummaryWriter

 
# config_params:
with open("config/config.yaml", "r") as file:
    config_data = yaml.safe_load(file)
data_path = config_data["paths"]["data_path"]
saved_models_path= config_data["paths"]["saved_models_path"]
saved_figures_path=config_data["paths"]["saved_figures_path"]
tensorboard_runs_path=config_data["paths"]["tensorboard_runs_path"]

#setup logger


#training params
LATENT_DIM=2
NUM_EPOCHS=50
LR=1e-02
BATCH_SIZE=256
SAMPLES_PER_EPOCH=60000
BATCHES_PER_EPOCH=int(SAMPLES_PER_EPOCH/BATCH_SIZE)
DROPOUT=0
UPPER_BOUND_LOGVAR=3
LOWER_BOUND_LOGVAR=-7
WEIGHT_DECAY=0.01
LR_GAMMA_DECAY=0.95

#system params
T_MAX=2
N_T=100
X0=0.5
XF=0.5

#setup
ACTION=HarmonicAction(T_max=T_MAX,N_T=N_T, x0=X0, xf=XF)
VAE_MODEL=VAE_FNN(latent_dim=LATENT_DIM, hidden_size=N_T-2, encoder_output_size=N_T-2, dropout=DROPOUT , upper_bound_logvar=UPPER_BOUND_LOGVAR, lower_bound_logvar=LOWER_BOUND_LOGVAR)
# VAE_MODEL=VAE_LSTM(latent_dim=LATENT_DIM, lstm_hidden_size=2, encoder_hidden_size=N_T-2, encoder_output_size=N_T-2, dropout=0 , upper_bound_logvar=3, lower_bound_logvar=-7)
# OPTIMIZER = optim.SGD(VAE_MODEL.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
OPTIMIZER=optim.Adam(VAE_MODEL.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
SCHEDULER = optim.lr_scheduler.ExponentialLR(OPTIMIZER, gamma=LR_GAMMA_DECAY)
LOSS_FUNC= GaussianElbo(action=ACTION)

#CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device {device}")
if device.type == "cuda":
    logging.info(torch.cuda.get_device_name(0))


params={
        "model_name": VAE_MODEL.__class__.__name__,
        "max_epochs": NUM_EPOCHS,
        "lr" : LR,
        "batch_size" : BATCH_SIZE,
        "batches_per_epoch":BATCHES_PER_EPOCH,
        "optimizer" : OPTIMIZER.__class__.__name__,
        "scheduler": SCHEDULER.__class__.__name__,
        "loss_func": LOSS_FUNC.__class__.__name__,
        "dropout": DROPOUT,
        "weight_decay":WEIGHT_DECAY,
        "lr_gamma_decay": LR_GAMMA_DECAY,
        "T_max":T_MAX,
        "N_T":N_T,
        "X0": X0,
        "XF": XF
        }

timestamp = str(datetime.now().strftime("%d-%m-%Y_%H-%M-%S"))
writer=SummaryWriter(log_dir=f"results/runs/run_{timestamp}")
writer.add_hparams(params,{})

def train(
        num_epochs,
        batches_per_epoch,
        batch_size,
        optimizer,
        scheduler,
        loss_func,
        model
        ):
    
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

        loss_vec.append(loss)
        avg_loss_vec.append(avg_train_loss)
        param_vec.append(params)

        print(100*"*")
        logging.info(f"finished epoch:{epoch}/{num_epochs}")
        logging.info(f"average train loss: {avg_train_loss}")
        logging.info(f"avg mu_x (along path) last batch: {torch.mean(params[0])}")
        logging.info(f"avg var_x (along path) last batch: {torch.mean(torch.exp(params[1]))}")

        writer.add_scalar("train_loss_avg_over_epoch",avg_train_loss, epoch)
        writer.add_scalar("mu_x vec_avg_over_batch",torch.mean(params[0]),epoch)
        writer.add_scalar("var_x vec_avg_over_batch",torch.mean(torch.exp(params[1])),epoch)

    return loss_vec, avg_loss_vec, param_vec

def main(
        num_epochs,
        batches_per_epoch,
        batch_size,
        optimizer,
        scheduler,
        loss_func,
        model,
        ):
    
    loss_vec, avg_loss_vec, param_vec=train(
                num_epochs= num_epochs,
                batches_per_epoch=batches_per_epoch,
                batch_size=batch_size,
                optimizer=optimizer,
                scheduler=scheduler,
                loss_func=loss_func,
                model=model
                )
    
    return loss_vec, avg_loss_vec, param_vec


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="description of script")
    parser.add_argument("--ARGUMENT", default="Hello World", type=str, help=f"ARGUMENT")
    args = parser.parse_args()


    loss_vec, avg_loss_vec, param_vec=main(
        num_epochs= NUM_EPOCHS,
        batches_per_epoch=BATCHES_PER_EPOCH,
        batch_size=BATCH_SIZE,
        optimizer=OPTIMIZER,
        scheduler=SCHEDULER,
        loss_func=LOSS_FUNC,
        model=VAE_MODEL
    )

 


 
 
