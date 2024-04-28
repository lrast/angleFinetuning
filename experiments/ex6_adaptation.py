# running through a few iterations of the adaptation process
import glob

from basicModel import EstimateAngle
from pytorch_lightning import Trainer
from adapt_fit_loop import adapt_fit_loop


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
