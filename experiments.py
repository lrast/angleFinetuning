import wandb
import glob
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
        runEarlyStoppingTraining(model, 
                                 directory=f'initialSweepResults/{dirName}')

    sweepid = wandb.sweep(sweep=sweepCFG, project='EstimateAngle')
    wandb.agent(sweepid, sweepRun)


def dataFile_initialSweep(kappa_tr, loc_tr, noiseVar, 
                          pixelDim, reps, shotNoise):
    dataDirectory = 'initialSweepResults/' + str(hash((kappa_tr, loc_tr,
                                                 noiseVar, pixelDim, reps,
                                                 shotNoise)))[1:]

    return glob.glob(dataDirectory+'/*')[0]


#  Experinment 2: A bit of a re-run of experiment 1, with a bug fix
#  knowing the right noise level
#  Goal: sensitivity scaling with 


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
                                 directory=f'Experiment2/{dirName}')

    ConcentratedSweepID = wandb.sweep(sweep=concentratedSweep,
                                      project='EstimateAngle')
    wandb.agent(ConcentratedSweepID, sweepRun)
    FlatSweepID = wandb.sweep(sweep=flatSweep, project='EstimateAngle')
    wandb.agent(FlatSweepID, sweepRun)


def dataFile_Experiment2(dataSize, kappa_tr, loc_tr, reps):
    dataDirectory = 'Experiment2/' + str(hash((dataSize, kappa_tr, 
                                              loc_tr, reps)))[1:]

    return glob.glob(dataDirectory+'/*')[0]
