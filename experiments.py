import wandb
import numpy as np

from basicModel import EstimateAngle
from trainers import runEarlyStoppingTraining


#  Experinment 1: sweeping across different noise levels and variances in the
#  training data.
#  Goal: find accuracy that depends on training mean


def InitialSweep_TrainDistributionAndNoise():
    sweepCFG = {
        'method': 'grid',
        'name': 'DistAndNoiseCharacterization',
        'parameters': {
            'reps': {'values': [0, 1]},
            'kappa_tr': {'values': [1., 2., 4.]},
            'loc_tr': {'values': (np.pi * np.linspace(0, 2, 7)).tolist()[0:2]},
            'shotNoise': {'values': [0.8]},
            'noiseVar': {'values': [1., 10., 20.]},
            'pixelDim': {'values': [101]}
        }
    }

    def sweepRun():
        """An individual training run for a sweep"""
        wandb.init()

        print(dict(wandb.config).values)

        dirName = str(hash(tuple(dict(wandb.config).values())))[1:]

        model = EstimateAngle(**wandb.config,
                              loc_val=wandb.config['loc_tr'],
                              kappa_val=wandb.config['kappa_tr'],
                              max_epochs=3000
                              )
        runEarlyStoppingTraining(model, directory=f'initialSweepResults/{dirName}')

    sweepid = wandb.sweep(sweep=sweepCFG, project='EstimateAngle')
    wandb.agent(sweepid, sweepRun)


def loadData_initialSweep(kappa_tr, loc_tr, noiseVar, 
                          pixelDim, reps, shotNoise):
    pass






