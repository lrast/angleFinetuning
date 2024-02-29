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


class Discriminator(pl.LightningModule):
    """Discriminator: trained for fine discrimination"""
    def __init__(self):
        super(Discriminator, self).__init__()
        self.save_hyperparameters({'lr': 1E-3})

        self.discriminationNet = nn.Sequential(
                    nn.Linear(2, 5),
                    nn.ReLU(),
                    nn.Linear(5, 5),
                    nn.ReLU(),
                    nn.Linear(5, 2),
                    nn.Softmax(dim=1)
            )
        self.lossFn = nn.BCELoss()

    def forward(self, X):
        likelihood = self.discriminationNet.forward(X)
        ls, inds = torch.max(likelihood, dim=1)

        return inds

    def training_step(self, batch, batchid=None):
        X, y = batch

        prediction = self.discriminationNet.forward(X)
        loss = self.lossFn(prediction, y)
        self.log('trainLoss', loss)
        return loss

    def validation_step(self, batch, batchid=None):
        X, y = batch

        probabilities = self.discriminationNet.forward(X)
        loss = self.lossFn(probabilities, y)

        predictions = self.forward(X)

        accuracy = y[torch.arange(y.shape[0]), predictions].sum() / y.shape[0]

        self.log('validAccuracy', accuracy)
        self.log('validLoss', loss)
        return accuracy

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


def make_dataloader(theta, dtheta, model_checkpoint, N_train=1000, N_val=500, batch_size=10):
    """ Generate samples from the models at hand """

    trained_model = EstimateAngle.load_from_checkpoint(model_checkpoint)
    X_train = trained_model.forward(
                            generateGrating(N_train//2*[theta] + N_train//2*[theta + dtheta],
                                            pixelDim=101, shotNoise=0.8, noiseVar=20)
                    ).detach()
    y_train = nn.functional.one_hot(
                        torch.tensor(N_train//2*[0.] + N_train//2*[1.]).long()
                    ).float()

    X_valid = trained_model.forward(
                            generateGrating(N_val//2*[theta] + N_val//2*[theta + dtheta],
                                            pixelDim=101, shotNoise=0.8, noiseVar=20)
                    ).detach()
    y_valid = nn.functional.one_hot(
                        torch.tensor(N_val//2*[0.] + N_val//2*[1.]).long()
                    ).float()

    trainDL = DataLoader(TensorDataset(X_train, y_train), shuffle=True, batch_size=batch_size)
    valDL = DataLoader(TensorDataset(X_valid, y_valid))

    return trainDL, valDL


def run_training(model, trainDL, validDL, directory, patience=200):
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
                      max_epochs=5000,
                      callbacks=[checkpoint_callback, earlystopping_callback]
                      )
    trainer.fit(model, trainDL, validDL)
