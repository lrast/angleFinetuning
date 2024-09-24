# simple neural network model for the task
import torch
import numpy as np
import pytorch_lightning as pl

from torch import nn

from torch.utils.data import DataLoader
from datageneration.stimulusGeneration import GratingDataset
from datageneration.faces.rotated_olivetti import FaceDataset
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
        self.testData = GratingDataset(testAngles, **gratingHyperparams)

    def train_dataloader(self):
        return DataLoader(self.trainingData,
                          batch_size=self.hparams.batchsize,
                          shuffle=True,
                          num_workers=2,
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
            del self.trainingData
            del self.valData
            del self.testData


class EstimateAngle_Faces(EstimateAngle):
    """EstimateAngle_Faces same as EstimateAngle above, but uses the faces dataset
    """
    def __init__(self, **kwargs):
        pixelDim = 64
        hyperparameterValues = {
            # stimulus generation hyperparameters
            'frequency': 0,
            'shotNoise': 0.,
            'noiseVar': 0.,
            'pixelDim': pixelDim,
            'seed': torch.random.seed(),
        }
        hyperparameterValues.update(kwargs)
        super(EstimateAngle_Faces, self).__init__(**hyperparameterValues)

        if 'hidden_dims' in kwargs:
            hidden_dims = kwargs['hidden_dims']
        else:
            hidden_dims = (pixelDim, 20)

        self.model = nn.Sequential(
            nn.Linear(pixelDim**2, hidden_dims[0]),
            nn.LeakyReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.LeakyReLU(),
            nn.Linear(hidden_dims[1], 2)
        )

    # redefine the angle encoding / decoding to for full 2pi degrees of encoding
    def encodeAngles(self, angles):
        return torch.stack((torch.cos(angles), torch.sin(angles)), dim=1)

    def decodeAngles(self, encodings):
        return torch.atan2(encodings[:, 1], encodings[:, 0])

    # data
    def setup(self, stage=None):
        """generate the datasets"""
        basesize = self.hparams.dataSize

        trainingAngles = vonmises(self.hparams.kappa_tr, self.hparams.loc_tr
                                  ).rvs(8*basesize)
        valAngles = vonmises(self.hparams.kappa_val, self.hparams.loc_val
                             ).rvs(2*basesize)
        testAngles = vonmises(self.hparams.kappa_test, self.hparams.loc_test
                              ).rvs(2*basesize)

        # generate datasets
        self.trainingData = FaceDataset(trainingAngles, split='train')
        self.valData = FaceDataset(valAngles, split='validation')
        self.testData = FaceDataset(testAngles, split='test')

    def validation_step(self, batch, batchidx):
        images = batch['image']
        targets = batch['angle']

        encodedTargets = self.encodeAngles(targets)
        prediction = self.forward(images)

        linear_loss_fn = lambda x, y: 1. - torch.mean(self.cosSim(x, y))
        linear_loss = linear_loss_fn(prediction, encodedTargets)

        self.log('Linear Val Loss', linear_loss.item())
        super(EstimateAngle_Faces, self).validation_step(batch, batchidx)


class EstimateAngle_Faces_final(EstimateAngle):
    """Uses architecture that is selected for well-behaved MSE curves.
    """
    def __init__(self, loss_pair, **kwargs):
        pixelDim = 64
        hidden_dims = (30, 10)

        hyperparameterValues = {
            # stimulus generation hyperparameters
            'frequency': 0,
            'shotNoise': 0.,
            'noiseVar': 0.,
            'pixelDim': pixelDim,
            'seed': torch.random.seed(),
            'max_epochs': 4000,
            'gradient_clip_val': 0.5,
            'loss_fn': loss_pair[0]
        }
        hyperparameterValues.update(kwargs)
        super(EstimateAngle_Faces_final, self).__init__(**hyperparameterValues)

        self.model = nn.Sequential(
            nn.Linear(pixelDim**2, hidden_dims[0]),
            nn.LeakyReLU(),
            nn.Dropout(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.LeakyReLU(),
            nn.Dropout(),
            nn.Linear(hidden_dims[1], 2)
        )

        # baseline similarity measure: higher is better
        cosSim = nn.CosineSimilarity()
        eps = 1E-8
        relu = nn.ReLU()

        self.positive_sim = lambda x, y: relu(cosSim(x,y) + 1 + eps/2) + eps/2
        self.lossFn = lambda x, y: loss_pair[1](self.positive_sim(x,y))


    # redefine the angle encoding / decoding to for full 2pi degrees of encoding
    def encodeAngles(self, angles):
        return torch.stack((torch.cos(angles), torch.sin(angles)), dim=1)

    def decodeAngles(self, encodings):
        return torch.atan2(encodings[:, 1], encodings[:, 0])

    # data
    def setup(self, stage=None):
        """generate the datasets"""
        basesize = self.hparams.dataSize

        trainingAngles = vonmises(self.hparams.kappa_tr, self.hparams.loc_tr
                                  ).rvs(8*basesize)
        valAngles = vonmises(self.hparams.kappa_val, self.hparams.loc_val
                             ).rvs(2*basesize)
        testAngles = vonmises(self.hparams.kappa_test, self.hparams.loc_test
                              ).rvs(2*basesize)

        # generate datasets
        self.trainingData = FaceDataset(trainingAngles, split='train')
        self.valData = FaceDataset(valAngles, split='validation')
        self.testData = FaceDataset(testAngles, split='test')

    def validation_step(self, batch, batchidx):
        """
        additionally logging the linear validation so that we can compare between
        different loss functions
        """
        images = batch['image']
        targets = batch['angle']

        encodedTargets = self.encodeAngles(targets)
        prediction = self.forward(images)

        linear_loss_fn = lambda x, y: 1. - torch.mean(self.cosSim(x, y))
        linear_loss = linear_loss_fn(prediction, encodedTargets)

        self.log('Linear Val Loss', linear_loss.item())
        super(EstimateAngle_Faces_final, self).validation_step(batch, batchidx)


    def teardown(self, stage):
        """ clearing the memory after it is used """
        if stage == "fit":
            del self.trainingData
            del self.valData
            del self.testData


class EstimateAngle_Faces_experimental(EstimateAngle):
    """ Class for experimenting with alternative model architectures
    """
    def __init__(self, loss_pair, **kwargs):
        pixelDim = 64
        if 'hidden_dims' in kwargs:
            hidden_dims = kwargs[hidden_dims]
        hidden_dims = (60, 20)

        hyperparameterValues = {
            # stimulus generation hyperparameters
            'frequency': 0,
            'shotNoise': 0.,
            'noiseVar': 0.,
            'pixelDim': pixelDim,
            'seed': torch.random.seed(),
            'max_epochs': 4000,
            'gradient_clip_val': 0.5,
            'loss_fn': loss_pair[0]
        }
        hyperparameterValues.update(kwargs)
        super(EstimateAngle_Faces_experimental, self).__init__(**hyperparameterValues)

        self.model = nn.Sequential(
            nn.Linear(pixelDim**2, hidden_dims[0]),
            nn.CELU(),
            nn.Dropout(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.CELU(),
            nn.Dropout(),
            nn.Linear(hidden_dims[1], 2)
        )

        # baseline similarity measure: higher is better
        cosSim = nn.CosineSimilarity()
        eps = 1E-8
        relu = nn.ReLU()

        self.positive_sim = lambda x, y: relu(cosSim(x,y) + 1 + eps/2) + eps/2
        self.lossFn = lambda x, y: loss_pair[1](self.positive_sim(x,y))


    # redefine the angle encoding / decoding to for full 2pi degrees of encoding
    def encodeAngles(self, angles):
        return torch.stack((torch.cos(angles), torch.sin(angles)), dim=1)

    def decodeAngles(self, encodings):
        return torch.atan2(encodings[:, 1], encodings[:, 0])

    # data
    def setup(self, stage=None):
        """generate the datasets"""
        basesize = self.hparams.dataSize

        trainingAngles = vonmises(self.hparams.kappa_tr, self.hparams.loc_tr
                                  ).rvs(8*basesize)
        valAngles = vonmises(self.hparams.kappa_val, self.hparams.loc_val
                             ).rvs(2*basesize)
        testAngles = vonmises(self.hparams.kappa_test, self.hparams.loc_test
                              ).rvs(2*basesize)

        # generate datasets
        self.trainingData = FaceDataset(trainingAngles, split='train')
        self.valData = FaceDataset(valAngles, split='validation')
        self.testData = FaceDataset(testAngles, split='test')

    def validation_step(self, batch, batchidx):
        """
        additionally logging the linear validation so that we can compare between
        different loss functions
        """
        images = batch['image']
        targets = batch['angle']

        encodedTargets = self.encodeAngles(targets)
        prediction = self.forward(images)

        linear_loss_fn = lambda x, y: 1. - torch.mean(self.cosSim(x, y))
        linear_loss = linear_loss_fn(prediction, encodedTargets)

        self.log('Linear Val Loss', linear_loss.item())
        super(EstimateAngle_Faces_experimental, self).validation_step(batch, batchidx)


    def teardown(self, stage):
        """ clearing the memory after it is used """
        if stage == "fit":
            del self.trainingData
            del self.valData
            del self.testData


class FaceAngle_Flat(EstimateAngle):
    """
        Face angle estimation with a limited ranges: what happends when the output is linear instead of circular?
    """
    def __init__(self, loss_pair, **kwargs):
        pixelDim = 64
        if 'hidden_dims' in kwargs:
            hidden_dims = kwargs['hidden_dims']
        else:
            hidden_dims = (60, 20)

        hyperparameterValues = {
            # stimulus generation hyperparameters
            'train_val_test_split': (256, 64, 80),
            'frequency': 0,
            'shotNoise': 0.,
            'noiseVar': 0.,
            'pixelDim': pixelDim,
            'seed': torch.random.seed(),
            # training hyperparameters
            'max_epochs': 4000,
            'gradient_clip_val': 0.5,
            'loss_fn': loss_pair[0],
        }
        hyperparameterValues.update(kwargs)
        super(FaceAngle_Flat, self).__init__(**hyperparameterValues)

        self.model = nn.Sequential(
            nn.Linear(pixelDim**2, hidden_dims[0]),
            nn.LeakyReLU(),
            #nn.Dropout(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.LeakyReLU(),
            #nn.Dropout(),
            nn.Linear(hidden_dims[1], 2),
            customTanh(np.pi/2 + 0.1, scaling=0.5)
        )

        # baseline distance measure: project onto the main diagonal, and measure distance there
        # lower is better
        eps = 1E-8
        def project(embeddings):
            unit_vector = torch.ones(2,1, device=embeddings.device) / 2**0.5
            return (embeddings @ unit_vector) @ unit_vector.T

        self.positive_diff = lambda x, y: torch.abs(project(x)[:,0] - project(y)[:,0])
        self.lossFn = lambda x, y: loss_pair[1](self.positive_diff(x,y))

    # now we are decoding the angle linearly along a diagonal line in embedding space
    def encodeAngles(self, angles):
        return (angles * torch.ones(2, angles.shape[0], device=angles.device)).T

    def decodeAngles(self, encodings):
        return (encodings  @ torch.ones(2,1, device=encodings.device)/2).T

    # data
    def setup(self, stage=None):
        """generate the datasets"""

        # for this model, we are explicitly training for only half of the interval,
        basesize = self.hparams.dataSize
        numbers = self.hparams.train_val_test_split

        trainingAngles = vonmises(self.hparams.kappa_tr, self.hparams.loc_tr
                                  ).rvs(8*basesize) / 2
        valAngles = vonmises(self.hparams.kappa_val, self.hparams.loc_val
                             ).rvs(2*basesize) / 2
        testAngles = vonmises(self.hparams.kappa_test, self.hparams.loc_test
                              ).rvs(2*basesize) / 2

        # generate datasets
        self.trainingData = FaceDataset(trainingAngles, split='train', numbers=numbers)
        self.valData = FaceDataset(valAngles, split='validation', numbers=numbers)
        self.testData = FaceDataset(testAngles, split='test', numbers=numbers)

    def validation_step(self, batch, batchidx):
        """
        additionally logging the linear validation so that we can compare between
        different loss functions
        """
        images = batch['image']
        targets = batch['angle']

        encodedTargets = self.encodeAngles(targets)
        prediction = self.forward(images)

        linear_loss = lambda d: torch.mean(d)

        self.log('Linear Val Loss', linear_loss(self.positive_diff(encodedTargets, prediction)).item())
        self.log('holder', 0) # hack for a strange wandb bug that I think is caused by logging too much

        super(FaceAngle_Flat, self).validation_step(batch, batchidx)


class customTanh(nn.Module):
    """custom Tanh non-linearity with a different range"""
    def __init__(self, multiplier, scaling=1):
        super(customTanh, self).__init__()
        self.multiplier = multiplier
        self.tanh = nn.Tanh()
        self.scaling = scaling

    def forward(self, x):
        return self.multiplier * self.tanh(self.scaling*x)
