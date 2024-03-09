# models and stimulus pdf for adaptation studies
import torch
import numpy as np

from torch.utils.data import DataLoader

from datageneration.stimulusGeneration import GratingDataset
from basicModel import EstimateAngle


class AngleDistribution(object):
    """AnglePDF: our discrete parameterized pdfs over angles"""
    def __init__(self, values):
        super(AngleDistribution, self).__init__()
        self.values = values
        self.npoints = len(values)

        # Linear interpolation between points, the sum is a trapezoid integral
        bin_probs = np.convolve(values, np.array([0.5, 0.5]), mode='valid')
        self.bin_probs = bin_probs / bin_probs.sum()

    def sample(self, N):
        """ Sample our distribution of angles """
        
        # for now, we will simply sample the midpoints of bins
        # improvement: also sample the linear interpolation between points
        # this would certainly help with smoothing

        bin_midpoints = np.convolve(np.linspace(-np.pi, np.pi, self.npoints),
                                    np.array([0.5, 0.5]), mode='valid')

        bin_inds = np.random.choice(range(self.npoints-1), size=(N,), p=self.bin_probs)

        return bin_midpoints[bin_inds]


class AdaptableEstimator(EstimateAngle):
    """
        Inheriting from the basic model, this model accepts different stimulus
        distributions. It can load parameters of a trained EstimateAngle model
    """
    def __init__(self, angle_dist, **kwargs):
        hyperparameterValues = {
            # training hyperparameters
            'lr': 1E-3,
            'batchsize': 32,
            'max_epochs': 3000,
            'dataSize': 512,
            # stimulus generation hyperparameters
            'frequency': 5,
            'shotNoise': 0.8,
            'noiseVar': 20.,
            'pixelDim': 101,
            'seed': torch.random.seed(),
        }
        hyperparameterValues.update(kwargs)

        super(AdaptableEstimator, self).__init__(**hyperparameterValues)

        for key in ['kappa_tr', 'kappa_val', 'kappa_test',
                    'loc_tr', 'loc_val', 'loc_test']:
            self.hparams.pop(key)

        self.angle_dist = angle_dist

        # set seeds
        torch.manual_seed(self.hparams.seed)
        np.random.seed(self.hparams.seed % (2**32-1))

    # data
    def setup(self, stage=None):
        """generate the datasets"""
        basesize = self.hparams.dataSize
        gratingHyperparams = {key: self.hparams[key] for key in 
                              ['frequency', 'shotNoise', 'noiseVar', 'pixelDim']
                              }

        trainingAngles = self.angle_dist.sample(8*basesize)
        valAngles = self.angle_dist.sample(2*basesize)

        # generate datasets
        self.trainingData = GratingDataset(trainingAngles, **gratingHyperparams)
        self.valData = GratingDataset(valAngles, **gratingHyperparams)
        # self.testData = GratingDataset(testAngles, **gratingHyperparams)

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
