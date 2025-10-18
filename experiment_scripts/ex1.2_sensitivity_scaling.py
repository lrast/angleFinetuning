#  Experiment 2: A bit of a re-run of experiment 1, with a bug fix
#  knowing the right noise level
#  Goal: sensitivity scaling

import wandb
import glob
import numpy as np

from basicModel import EstimateAngle
from trainers import runEarlyStoppingTraining


def Experiment2_trainingSetSize():
    """ What is the impact of training on distributions concentrated
        on different means?
        How is this modified by  
    """

    defaultConfig = {
                     'pixelDim': 101,
                     'shotNoise': 0.8,
                     'noiseVar': 20.
                     }

    concentratedSweep = {
        'method': 'grid',
        'name': 'Exp2concentrated',
        'parameters': {
            'reps': {'values': [0, 1]},
            'loc_tr': {'values': [0., np.pi/3]},
            'kappa_tr': {'values': [4.]},
            'dataSize': {'values': [512, 1024, 2048]}
        }
    }

    flatSweep = {
        'method': 'grid',
        'name': 'Exp2flat',
        'parameters': {
            'reps': {'values': [0, 1]},
            'loc_tr': {'values': [0.]},
            'kappa_tr': {'values': [1E-16]},
            'dataSize': {'values': [512, 1024, 2048]}
        }
    }

    def sweepRun():
        """An individual training run for a sweep"""
        wandb.init()

        print(dict(wandb.config))

        dirName = str(hash(tuple(dict(wandb.config).values())))[1:]

        model = EstimateAngle(**defaultConfig,
                              **wandb.config,
                              loc_val=wandb.config['loc_tr'],
                              kappa_val=wandb.config['kappa_tr'],
                              max_epochs=3000
                              )
        runEarlyStoppingTraining(model, 
                                 directory=f'trainedParameters/Experiment2/{dirName}')

    ConcentratedSweepID = wandb.sweep(sweep=concentratedSweep,
                                      project='EstimateAngle')
    wandb.agent(ConcentratedSweepID, sweepRun)
    FlatSweepID = wandb.sweep(sweep=flatSweep, project='EstimateAngle')
    wandb.agent(FlatSweepID, sweepRun)


def dataFile_Experiment2(dataSize, kappa_tr, loc_tr, reps):
    dataDirectory = 'trainedParameters/Experiment2/' + str(hash((dataSize, kappa_tr, 
                                                           loc_tr, reps)))[1:]

    return glob.glob(dataDirectory+'/*')[0]
