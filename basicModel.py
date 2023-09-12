# simple neural network model for the task
import torch
import numpy as np
import pytorch_lightning as pl

from torch import nn

from torch.utils.data import DataLoader
from datageneration.stimulusGeneration import GratingDataset
from scipy.stats import vonmises


class EstimateAngle(pl.LightningModule):
    """Neural network models to estimate angle"""
    def __init__(self, **kwargs):
        super(EstimateAngle, self).__init__()

        hyperparameterValues = {
            # training hyperparameters
            'lr': 1E-3,
            'batchsize': 32,
            'max_epochs': 40,
            'dataSize': 1024,
            # stimulus generation hyperparameters
            'frequency': 5,
            'shotNoise': 0.,
            'noiseVar': 0.,
            'pixelDim': 201,
            'seed': torch.random.seed(),
            # angle distribution hyperparameters
            'kappa_tr': 1.,
            'loc_tr': 0.,
            'kappa_val': 1.,
            'loc_val': 0.,
            'kappa_test': 1.,
            'loc_test': 0.
        }
        hyperparameterValues.update(kwargs)
        self.save_hyperparameters(hyperparameterValues)

        # set seeds
        torch.manual_seed(self.hparams.seed)
        np.random.seed(self.hparams.seed % (2**32-1))

        # make model
        pixelDim = self.hparams.pixelDim

        self.model = nn.Sequential(
                nn.Linear(pixelDim**2, pixelDim),
                nn.ReLU(),
                nn.Linear(pixelDim, 20),
                nn.ReLU(),
                nn.Linear(20, 2)
        )

        self.cosSim = nn.CosineSimilarity()
        self.lossFn = lambda x, y: torch.mean(1. - self.cosSim(x, y))

    # angle coding functions
    def encodeAngles(self, angles):
        return torch.stack((torch.cos(2*angles), torch.sin(2*angles)), dim=1)

    def decodeAngles(self, encodings):
        return torch.atan2(encodings[:, 1], encodings[:, 0]) / 2.

    def forward(self, images):
        nsamples = images.shape[0]
        return self.model(images.view(nsamples, -1))

    def training_step(self, batch, batchidx):
        images = batch['image']
        targets = batch['angle']

        encodedTargets = self.encodeAngles(targets)
        prediction = self.forward(images)

        loss = self.lossFn(prediction, encodedTargets)
        self.log('Train Loss', loss.item())

        return loss

    def validation_step(self, batch, batchidx):
        images = batch['image']
        targets = batch['angle']

        encodedTargets = self.encodeAngles(targets)
        prediction = self.forward(images)

        loss = self.lossFn(prediction, encodedTargets)
        self.log('Val Loss', loss.item())

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    # data
    def setup(self, stage=None):
        """generate the datasets"""
        basesize = self.hparams.dataSize
        gratingHyperparams = {key: self.hparams[key] for key in 
                              ['frequency', 'shotNoise', 'noiseVar', 'pixelDim']
                              }

        trainingAngles = vonmises(self.hparams.kappa_tr, self.hparams.loc_tr
                                  ).rvs(8*basesize) % np.pi
        valAngles = vonmises(self.hparams.kappa_val, self.hparams.loc_val
                             ).rvs(2*basesize) % np.pi
        testAngles = vonmises(self.hparams.kappa_test, self.hparams.loc_test
                              ).rvs(2*basesize) % np.pi

        # generate datasets
        self.trainingData = GratingDataset(trainingAngles, **gratingHyperparams)
        self.valData = GratingDataset(valAngles, **gratingHyperparams)
        # self.testData = GratingDataset(testAngles, **gratingHyperparams)

    def train_dataloader(self):
        return DataLoader(self.trainingData,
                          batch_size=self.hparams.batchsize,
                          shuffle=True,
                          num_workers=2
                          )

    def val_dataloader(self):
        return DataLoader(self.valData, batch_size=2*self.hparams.dataSize,
                          num_workers=2
                          )

    def test_dataloader(self):
        return DataLoader(self.testData, batch_size=self.hparams.batchsize)
