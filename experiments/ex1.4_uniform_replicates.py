
import wandb
import glob

import numpy as np

from basicModel import EstimateAngle
from trainers import runEarlyStoppingTraining

from pytorch_lightning import Trainer


def Experiment4_baselines():
    """ Baseline untrained and trained models, uniform stimuli
    """
    defaultConfig = {
                     'pixelDim': 101,
                     'shotNoise': 0.8,
                     'noiseVar': 20.,
                     'dataSize': 512,
                     'loc_tr': 0.,
                     'kappa_tr': 1E-16,
                     'loc_val': 0.,
                     'kappa_val': 1E-6
                     }

    for rep in range(6):
        """An individual training run for a sweep"""
        wandb.init(reinit=True)
        print('################ rep #############')
        model = EstimateAngle(**defaultConfig,
                              max_epochs=3000
                              )

        dummy = Trainer(max_epochs=0)
        dummy.fit(model)
        dummy.save_checkpoint(f'trainedParameters/Exp4/rep{rep}/pretrain.ckpt')

        runEarlyStoppingTraining(model, 
                                 directory=f'trainedParameters/Exp4/rep{rep}/')
                                 

def Experiment4_concentrated():
    """ Baseline trained models, non-uniform stimuli
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

    for rep in range(6):
        """An individual training run for a sweep"""
        wandb.init(reinit=True)
        print('################ rep #############')
        model = EstimateAngle(**defaultConfig,
                              max_epochs=3000
                              )

        runEarlyStoppingTraining(model,
                                 directory=f'trainedParameters/Exp4_conc/rep{rep}/')

