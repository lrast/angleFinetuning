# simple neural network model for the task
import torch
import numpy as np
import pytorch_lightning as pl

from torch import nn

from torch.utils.data import DataLoader
from datageneration.stimulusGeneration import GratingDataset
from scipy.stats import vonmises


class estimateAngle(pl.LightningModule):
    """Neural network models to estimate angle"""
    def __init__(self, **kwargs):
        super(estimateAngle, self).__init__()

        self.hyperparameters = {
            # training hyperparameters
            'lr': 1E-3,
            'batchsize': 100,
            'max_epochs': 100,
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
        self.hyperparameters.update(kwargs)

        # set seeds
        torch.manual_seed(self.hyperparameters['seed'])
        np.random.seed( self.hyperparameters['seed'] % (2**32-1) )

        # make model
        pixelDim = self.hyperparameters['pixelDim']

        self.model = nn.Sequential(
                nn.Linear( pixelDim**2, pixelDim),
                nn.ReLU(),
                nn.Linear( pixelDim, 20),
                nn.ReLU(),
                nn.Linear(20, 1)
        )

        self.lossFn = nn.MSELoss()

    def forward(self, angles):
        nsamples = angles.shape[0]
        return self.model( angles.view(nsamples, -1) )

    def training_step(self, batch, batchidx):
        images = batch['image']
        targets = batch['angle']

        loss = self.lossFn( self.forward(images).view(-1), targets )
        self.log('Train Loss', loss.detach())

        return loss


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hyperparameters['lr'])


    # data
    def setup(self, stage=None, basesize=1000):
        """generate the datasets"""
        gratingHyperparams = {key: self.hyperparameters[key] for key in ['frequency', 'shotNoise', 'noiseVar', 'pixelDim']}

        trainingAngles = vonmises(
            self.hyperparameters['kappa_tr'], self.hyperparameters['loc_tr']
            ).rvs(8*basesize) % np.pi
        valAngles = vonmises(
            self.hyperparameters['kappa_val'], self.hyperparameters['loc_val']
            ).rvs(2*basesize) % np.pi
        testAngles = vonmises(
            self.hyperparameters['kappa_test'], self.hyperparameters['loc_test']
            ).rvs(2*basesize) % np.pi

        # generate datasets
        self.trainingData = GratingDataset( trainingAngles, **gratingHyperparams)
        self.valData = GratingDataset( valAngles, **gratingHyperparams)
        self.testData = GratingDataset( testAngles, **gratingHyperparams)


    def train_dataloader(self):
        return DataLoader(self.trainingData, batch_size=self.hyperparameters['batchsize'])

    def val_dataloader(self):
        return DataLoader(self.valData, batch_size=self.hyperparameters['batchsize'])

    def test_dataloader(self):
        return DataLoader(self.testData, batch_size=self.hyperparameters['batchsize'])



