# splitting out the working models that we will focus on.

import torch
import numpy as np
import pytorch_lightning as pl

from torch import nn


from torch.utils.data import DataLoader
from datageneration.faces.rotated_olivetti import FaceDataset


class FaceAngle(pl.LightningModule):
    """
        General Face angle estimation.

        I'm trying to avoid too much fiddling with network parameters.
        I also want to avoid dropout if at all possible.
    """
    def __init__(self, loss_pair, distribution_pair, encoding, **kwargs):
        super(FaceAngle, self).__init__()
        hyperparameterValues = {
            # architecture hyperparameters
            'hidden_dims': (60, 20),
            # training hyperparameters
            'lr': 1E-3,
            'batchsize': 32,
            'max_epochs': 1500,
            'dataSize': 1024,
            'gradient_clip_val': 0.5,
            'loss_pair': loss_pair,
            # stimulus generation hyperparameters
            'train_val_test_split': (300, 50, 50),
            'pixelDim': 64,
            'seed': torch.random.seed(),
            # angle distribution hyperparameters
            'distribution_pair': distribution_pair,
            'encoding': encoding
        }
        hyperparameterValues.update(kwargs)
        self.save_hyperparameters(hyperparameterValues)

        self.get_angle_samples = distribution_pair[1]

        # set seeds
        torch.manual_seed(self.hparams.seed)
        np.random.seed(self.hparams.seed % (2**32-1))

        self.model = nn.Sequential(
            nn.Linear(self.hparams.pixelDim**2, self.hparams.hidden_dims[0]),
            nn.CELU(),
            nn.Linear(self.hparams.hidden_dims[0], self.hparams.hidden_dims[1]),
            nn.CELU(),
            nn.Linear(self.hparams.hidden_dims[1], 2)
        )

        eps = 1E-8

        loss = loss_pair[1]
        if encoding == 'linear':
            # baseline distance measure: project onto the main diagonal, and measure
            # lower is better
            def project(embeddings):
                unit_vector = torch.ones(2, 1) / 2**0.5
                unit_vector = unit_vector.to(embeddings.device)
                return (embeddings @ unit_vector) @ unit_vector.T

            def positive_diff(x, y):
                return torch.abs(project(x)[:, 0] - project(y)[:, 0]) + eps

            self.lossFn = lambda x, y: torch.mean(loss(positive_diff(x, y)))
            self.linear_loss = lambda x, y: torch.mean(positive_diff(x, y))

        if encoding == 'circular':
            # decode based on orientation relative to the origin
            cosSim = nn.CosineSimilarity()
            relu = nn.ReLU()

            def positive_sim(x, y):
                return relu(cosSim(x, y) + 1 + eps/2) + eps / 2

            self.lossFn = lambda x, y: loss(torch.tensor(2.)) - torch.mean(loss(positive_sim(x, y)))
            self.linear_loss = lambda x, y: 2. - torch.mean(positive_sim(x, y))

    def encodeAngles(self, angles):
        if self.hparams.encoding == 'linear':
            return (angles * torch.ones(2, angles.shape[0], device=angles.device)).T
        if self.hparams.encoding == 'circular':
            return torch.stack((torch.cos(angles), torch.sin(angles)), dim=1)

    def decodeAngles(self, encodings):
        if self.hparams.encoding == 'linear':
            return (encodings  @ torch.ones(2, 1, device=encodings.device)/2).T
        if self.hparams.encoding == 'circular':
            return torch.atan2(encodings[:, 1], encodings[:, 0])

    def forward(self, images):
        """ run the network """
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

        linear_loss = self.linear_loss(prediction, encodedTargets)

        self.log_dict({'Val Loss': loss.item(),
                       'Linear Val Loss': linear_loss.item()})

    def test_step(self, batch, batchidx):
        images = batch['image']
        targets = batch['angle']

        encodedTargets = self.encodeAngles(targets)
        prediction = self.forward(images)

        loss = self.lossFn(prediction, encodedTargets)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    # data
    def setup(self, stage=None):
        """generate the datasets"""
        basesize = self.hparams.dataSize
        numbers = self.hparams.train_val_test_split

        trainAngles = self.get_angle_samples(8*basesize)
        valAngles = self.get_angle_samples(2*basesize)
        testAngles = self.get_angle_samples(2*basesize)

        # generate datasets
        self.trainData = FaceDataset(trainAngles, split='train', numbers=numbers)
        self.valData = FaceDataset(valAngles, split='validation', numbers=numbers)
        self.testData = FaceDataset(testAngles, split='test', numbers=numbers)

    def train_dataloader(self):
        return DataLoader(self.trainData,
                          batch_size=self.hparams.batchsize,
                          shuffle=True,
                          num_workers=8,
                          persistent_workers=True
                          )

    def val_dataloader(self):
        return DataLoader(self.valData, batch_size=2*self.hparams.dataSize,
                          num_workers=2,
                          persistent_workers=True
                          )

    def test_dataloader(self):
        return DataLoader(self.testData, batch_size=2*self.hparams.dataSize)

    def teardown(self, stage):
        """ clearing the memory after it is used """
        if stage == "fit":
            del self.trainData
            del self.valData
            del self.testData
