
import wandb
import glob

import numpy as np

from basicModel import EstimateAngle
from trainers import runEarlyStoppingTraining

import torch
from torch.nn import CosineSimilarity

def Experiment7_loss_functions():
    """ Training models with different loss functions wrapping the cosine similarity:
        Do differences in the outputs emerge?
    """
    defaultConfig = {
                     'pixelDim': 101,
                     'shotNoise': 0.8,
                     'noiseVar': 20.,
                     'dataSize': 512,
                     'loc_tr': np.pi/2,
                     'kappa_tr': 8.,
                     'loc_val': np.pi/2,
                     'kappa_val': 8.
                     }

    # Cosine similarity is large (close to one) if the two vectors are
    # similar to each other
    cosSim = CosineSimilarity()
    eps = 1E-16

    # off-set to ensure that the cos similarity is always positive

    for rep in range(4):
        """An individual training run for a sweep"""
        print(f'################ rep {rep} #############')


        model = EstimateAngle(**defaultConfig,
                              max_epochs=3000
                              )
        loss = lambda x, y: 1.-torch.mean(cosSim(x, y))
        model.lossFn = loss

        runEarlyStoppingTraining(model,
                                 directory=f'trainedParameters/Exp7/linear/rep{rep}/')

        model2 = EstimateAngle(**defaultConfig,
                              max_epochs=3000
                              )
        loss = lambda x, y: np.log(2.)-torch.mean(torch.log(cosSim(x, y) + 1. + eps))
        model2.lossFn = loss

        runEarlyStoppingTraining(model2,
                                 directory=f'trainedParameters/Exp7/log/rep{rep}/')

        model3 = EstimateAngle(**defaultConfig,
                              max_epochs=3000
                              )
        loss = lambda x, y: np.sqrt(2)-torch.mean(torch.sqrt(cosSim(x, y) + 1. + eps))
        model3.lossFn = loss

        runEarlyStoppingTraining(model3,
                                 directory=f'trainedParameters/Exp7/sqrt/rep{rep}/')


def Experiment7_loss_functions_uniform():
    """ Training models with different loss functions wrapping the cosine similarity:
        Do differences in the outputs emerge?
    """
    defaultConfig = {
                     'pixelDim': 101,
                     'shotNoise': 0.8,
                     'noiseVar': 20.,
                     'dataSize': 512,
                     'loc_tr': np.pi/2,
                     'kappa_tr': 1E-8,
                     'loc_val': np.pi/2,
                     'kappa_val': 1E-8
                     }

    # Cosine similarity is large (close to one) if the two vectors are
    # similar to each other
    cosSim = CosineSimilarity()
    eps = 1E-16

    # off-set to ensure that the cos similarity is always positive

    for rep in range(4):
        """An individual training run for a sweep"""
        print(f'################ rep {rep} #############')


        model = EstimateAngle(**defaultConfig,
                              max_epochs=3000
                              )
        loss = lambda x, y: 1.-torch.mean(cosSim(x, y))
        model.lossFn = loss

        runEarlyStoppingTraining(model,
                                 directory=f'trainedParameters/Exp7/uniform/linear/rep{rep}/')

        model2 = EstimateAngle(**defaultConfig,
                              max_epochs=3000
                              )
        loss = lambda x, y: np.log(2.)-torch.mean(torch.log(cosSim(x, y) + 1. + eps))
        model2.lossFn = loss

        runEarlyStoppingTraining(model2,
                                 directory=f'trainedParameters/Exp7/uniform/log/rep{rep}/')

        model3 = EstimateAngle(**defaultConfig,
                              max_epochs=3000
                              )
        loss = lambda x, y: np.sqrt(2)-torch.mean(torch.sqrt(cosSim(x, y) + 1. + eps))
        model3.lossFn = loss

        runEarlyStoppingTraining(model3,
                                 directory=f'trainedParameters/Exp7/uniform/sqrt/rep{rep}/')
