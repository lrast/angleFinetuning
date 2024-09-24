from trainers import trainModel
from basicModel import EstimateAngle_Faces_experimental
from parameters import uniformConfig, concentratedConfig_1, concentratedConfig_2, lossFns

import torch

def Experiment21_CELU_0():
    """ The Fisher information curves that we fit for ReLU models were quite variable.
    I think that this is a result of the non-differentiability of the objective network.
    Trying the same experiment with a differentiable non-linearity
    """

    distParams = {
        'uniform': uniformConfig,
        'high_conc': concentratedConfig_1,
        'low_conc': concentratedConfig_2
    }

    hidden_dims = (30, 10)
    pixelDim = 64

    for rep in range(3):
        for loss_pair in lossFns.items():
            for dist_name in distParams.keys():
                model = EstimateAngle_Faces_experimental(loss_pair, **distParams[dist_name])

                trainModel(model, f'trainedParameters/Exp10/bigCELU/{loss_pair[0]}/{dist_name}/rep{rep}/')
