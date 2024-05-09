# running through a few iterations of the adaptation process
import glob
import torch
import numpy as np
import pandas as pd

from pytorch_lightning import Trainer

from basicModel import EstimateAngle
from adapt_fit_loop import adapt_fit_loop
from trainers import trainEarlyStoppingAndLoad, trainEarlyStoppingAndLoad_customTest
from discriminationAnalysis import Fisher_smooth_fits


def adaptation_experiment():
    # first, look into the adpatation of an untrained network

    defaultConfig = {
                     'pixelDim': 101,
                     'shotNoise': 0.8,
                     'noiseVar': 20.,
                     'dataSize': 512
                     }

    model = EstimateAngle(**defaultConfig,
                          max_epochs=3000
                          )

    dummy = Trainer(max_epochs=0)
    dummy.fit(model)

    # checkpoints 
    untrained = 'trainedParameters/Exp6/untrained.ckpt'

    dummy.save_checkpoint(untrained)

    checkpoints = {'untrained': untrained,
                   'uniform': glob.glob('trainedParameters/Exp4/rep0/epoch*')[0],
                   'concentrated': glob.glob('trainedParameters/Exp4_conc/rep0/epoch*')[0]}

    for name, checkpoint in checkpoints.items():
        to_save = f'trainedParameters/Exp6/{name}'
        adapt_fit_loop(checkpoint, to_save, max_iter=7)


def initialization_effect():
    # how big is the impact of initialization on the learned results?

    defaultConfig = {
                     'pixelDim': 101,
                     'shotNoise': 0.8,
                     'noiseVar': 20.,
                     'dataSize': 512
                     }

    row_data = []

    for rep in range(6):
        directory = f'trainedParameters/Exp6_init/rep{rep}/'

        init_model = EstimateAngle(**defaultConfig, max_epochs=1000)

        init_model.save_checkpoint(directory + 'initialized.ckpt')
        for retrained in range(3):
            model = EstimateAngle.load_from_checkpoint(directory + 'initialized.ckpt')

            trainEarlyStoppingAndLoad(model, directory + f'retrained{retrained}')
            fi = Fisher_smooth_fits(model, 0., np.pi, N_mean=10000, N_cov=500, Samp_cov=500)

            row = {'rep': rep, 'method': 'loaded', 'Fisher': fi}
            row_data.append(row)

        for state_update in range(3):
            model = EstimateAngle(**defaultConfig, max_epochs=1000)
            model.load_state_dict(init_model.state_dict())

            trainEarlyStoppingAndLoad(model, directory + f'state_update{retrained}')
            fi = Fisher_smooth_fits(model, 0., np.pi, N_mean=10000, N_cov=500, Samp_cov=500)

            row = {'rep': rep, 'method': 'set state', 'Fisher': fi}
            row_data.append(row)

    pd.DataFrame(row_data).to_pickle('experiment_result/ex6_initialization.pickle')


def initialization_effect_2():
    """
        The objective of this experiment is to determine whether saving only the weights
        is sufficient to remove the Fisher information correlations that result from
        pretraining.
        Essentially, do we have to search more for the origin of trajectory correlations?
    """

    defaultConfig = {
                     'pixelDim': 101,
                     'shotNoise': 0.8,
                     'noiseVar': 20.,
                     'dataSize': 512
                     }

    preTrainConfig = {
                     'loc_tr': np.pi/2,
                     'kappa_tr': 8.,
                     'loc_val': np.pi / 2,
                     'kappa_val': 8.
                    }

    postTrainConfig = {
                     'loc_tr': 0.0,
                     'kappa_tr': 0.1,
                     'loc_val': 0.0,
                     'kappa_val': 0.1
                    }

    row_data = []

    def get_Fisher(model): return Fisher_smooth_fits(model, 0., np.pi, N_mean=10000,
                                                     N_cov=500, Samp_cov=500)

    for rep in range(3):
        for pretrain in ['weights', 'all']:
            directory = f'trainedParameters/Exp6_init_2/pre{pretrain}/rep{rep}/'

            init_model = EstimateAngle(**defaultConfig, **preTrainConfig,
                                       max_epochs=1000)

            weights_only = (True if pretrain == 'weights' else False)

            init_ckpt = trainEarlyStoppingAndLoad_customTest(init_model, directory+'init/',
                                                             save_weights_only=weights_only)
            fi = get_Fisher(init_model)

            row = {'rep': rep, 'method': 'init', 'Fisher': fi}
            row_data.append(row)

            for retrained in range(6):
                model = EstimateAngle.load_from_checkpoint(init_ckpt, **postTrainConfig,
                                                           seed=torch.random.seed())

                trainEarlyStoppingAndLoad(model, directory + f'retrained{retrained}')
                fi = get_Fisher(model)

                row = {'rep': rep, 'method': pretrain, 'Fisher': fi}
                row_data.append(row)

    pd.DataFrame(row_data).to_pickle('experiment_result/ex6_init2.pickle')
