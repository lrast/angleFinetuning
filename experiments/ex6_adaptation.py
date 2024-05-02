# running through a few iterations of the adaptation process
import glob
import numpy as np
import pandas as pd

from pytorch_lightning import Trainer

from basicModel import EstimateAngle
from adapt_fit_loop import adapt_fit_loop
from trainers import trainEarlyStoppingAndLoad
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
    directory = 'trainedParameters/Exp6_'

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
