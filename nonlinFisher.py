# one way to nail down the Fisher information is to treat it as a fine discrimination task
# and use a general decoder to determine the outputs

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import pytorch_lightning as pl

from basicModel import EstimateAngle
from datageneration.stimulusGeneration import generateGrating

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

import wandb

from discriminationAnalysis import generate_samples

import pandas as pd
import numpy as np


def test_script():
    """ Running our Fisher information test on a pretrained model """
    ckpt = 'trainedParameters/Experiment2/380931818027565204/epoch=208-step=26752.ckpt'
    model = EstimateAngle.load_from_checkpoint(ckpt)

    stims = []
    FIs = []
    
    for theta in [0., np.pi/4, np.pi/2, 3*np.pi/4]:
        for rep in range(3):
            FIs.append(Fisher_nlin_decoder(model, theta, 0.05).item())
            stims.append(theta)

            print(theta)

    return pd.DataFrame({'Fisher': FIs, 'stimulus': stims})


def Fisher_nlin_decoder(model, theta, dtheta, N_train=5000, N_val=1000, N_test=1000):
    """EXPENSIVE!
       First, train a nonlinear decoder, then use that to evaluate
       the discrimination performance
     """
    discriminator = NeuralDecoder()
    traindl, validdl = make_dataloader(theta, dtheta, trained_model=model,
                                       N_train=N_train, N_val=N_val)

    best_discriminator = run_training(discriminator, traindl, validdl, './FIruns')

    thetas0 = N_test * [theta - dtheta/2]
    thetas1 = N_test * [theta + dtheta/2]

    left_decoding = best_discriminator(torch.tensor(
                                         generate_samples(model, thetas0))
                                       )
    right_decoding = best_discriminator(torch.tensor(
                                         generate_samples(model, thetas1))
                                        )

    FI = ((right_decoding.mean() - left_decoding.mean()) / dtheta)**2 /\
         (0.5 * (right_decoding.var() + left_decoding.var()))

    return FI


class NeuralDecoder(pl.LightningModule):
    """NeuralDecoder: trained for decoding neural activity data"""
    def __init__(self):
        super(NeuralDecoder, self).__init__()
        self.save_hyperparameters({'lr': 1E-3})

        self.discriminationNet = nn.Sequential(
                    nn.Linear(2, 5),
                    nn.ReLU(),
                    nn.Linear(5, 5),
                    nn.ReLU(),
                    nn.Linear(5, 1),
            )
        self.lossFn = nn.MSELoss()

    def forward(self, X):
        angles = self.discriminationNet.forward(X)

        return angles

    def training_step(self, batch, batchid=None):
        X, y = batch

        prediction = self.forward(X)
        loss = self.lossFn(prediction, y)
        self.log('trainLoss', loss)
        return loss

    def validation_step(self, batch, batchid=None):
        X, y = batch

        angles = self.forward(X)
        loss = self.lossFn(angles, y)

        self.log('validLoss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


def make_dataloader(theta, dtheta, model_checkpoint=None, 
                    trained_model=None,
                    N_train=1000, N_val=500, batch_size=10):
    """ Generate samples from the models at hand """

    if trained_model is None:
        trained_model = EstimateAngle.load_from_checkpoint(model_checkpoint)

    train_angles = torch.linspace(theta - dtheta, theta + dtheta, N_train) 

    X_train = trained_model.forward(
                            generateGrating(train_angles,
                                            pixelDim=101, shotNoise=0.8, noiseVar=20)
                    ).detach()
    y_train = train_angles[:, None]

    valid_angles = theta - dtheta + 2*dtheta*torch.rand(N_val)
    X_valid = trained_model.forward(
                            generateGrating(valid_angles,
                                            pixelDim=101, shotNoise=0.8, noiseVar=20)
                    ).detach()
    y_valid = valid_angles[:, None]

    trainDL = DataLoader(TensorDataset(X_train, y_train), shuffle=True,
                         batch_size=batch_size)
    valDL = DataLoader(TensorDataset(X_valid, y_valid), batch_size=20, num_workers=1,
                       persistent_workers=True)

    return trainDL, valDL


def run_training(model, trainDL, validDL, directory, patience=50):
    """Simple training behavior with checkpointing"""
    wandb.init(reinit=True, project='Fisher Info')
    wandb_logger = WandbLogger()

    earlystopping_callback = EarlyStopping(monitor='validLoss', mode='min', 
                                           patience=patience
                                           )
    checkpoint_callback = ModelCheckpoint(dirpath=directory,
                                          every_n_epochs=1, 
                                          save_top_k=1,
                                          monitor='validLoss'
                                          )

    trainer = Trainer(logger=wandb_logger,
                      max_epochs=2000,
                      callbacks=[checkpoint_callback, earlystopping_callback]
                      )
    trainer.fit(model, trainDL, validDL)

    return NeuralDecoder.load_from_checkpoint(checkpoint_callback.best_model_path)
