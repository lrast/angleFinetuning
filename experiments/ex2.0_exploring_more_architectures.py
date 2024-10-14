from trainers import trainModel
from basicModel import EstimateAngle_Faces_experimental
from parameters import uniformConfig, concentratedConfig_1, \
                       concentratedConfig_2, lossFns

import torch

from workingModels import FaceAngle
from scipy.stats import vonmises, norm, uniform


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

    for rep in range(3):
        for loss_pair in lossFns.items():
            for dist_name in distParams.keys():
                model = EstimateAngle_Faces_experimental(loss_pair, **distParams[dist_name])

                trainModel(model, f'trainedParameters/Exp10/bigCELU/{loss_pair[0]}/{dist_name}/rep{rep}/')


def Exp22_sweep_linear_and_circular():
    """
        I've now fixed a model architecture. Running a couple of model pairs.
    """
    losses = [('linear', torch.nn.Identity()), ('sqrt', torch.sqrt)]
    dists = {
            'linear': [('uniform', uniform(-torch.pi/2, torch.pi).rvs),
                       ('concentrated', norm(torch.pi/4, 0.2).rvs)
                       ],
            'circular': [('uniform', uniform(-torch.pi, 2*torch.pi).rvs),
                         ('concentrated', vonmises(torch.pi/2, 2.).rvs)
                         ]
            }

    for loss_pair in losses:
        for rep in range(3):
            for encoding in ['linear', 'circular']:
                for distribution_pair in dists[encoding]:
                    model = FaceAngle(loss_pair, distribution_pair, encoding)
                    trainModel(model,
                               f'trainedParameters/Exp2.2/{loss_pair[0]}_loss/{encoding}_enc/{distribution_pair[0]}/rep{rep}',
                               save_best_train=True
                               )


def Exp23_sweep_3():
    """
        I need more aggressive loss function changes to detect et 
    """
    losses = [('square', lambda x: x**2),
              ('quart', lambda x: x**4)
              ]
    dists = {
            'linear': [('uniform', uniform(-torch.pi/2, torch.pi).rvs),
                       ('concentrated', norm(torch.pi/4, 0.2).rvs)
                       ],
            'circular': [('uniform', uniform(-torch.pi, 2*torch.pi).rvs),
                         ('concentrated', vonmises(torch.pi/2, 2.).rvs)
                         ]
            }

    for loss_pair in losses:
        for rep in range(3):
            for encoding in ['linear', 'circular']:
                for distribution_pair in dists[encoding]:
                    model = FaceAngle(loss_pair, distribution_pair, encoding)
                    trainModel(model,
                               f'trainedParameters/Exp2.2/{loss_pair[0]}_loss/{encoding}_enc/{distribution_pair[0]}/rep{rep}',
                               save_best_train=True
                               )


def Exp23_sweep_losses():
    """
        I need more aggressive loss function changes to detect et 
    """
    losses = [('root4', lambda x: x**0.25),
              ('log', torch.log)
              ]
    dists = {
            'linear': [('uniform', uniform(-torch.pi/2, torch.pi).rvs),
                       ('concentrated', norm(torch.pi/4, 0.2).rvs)
                       ],
            'circular': [('uniform', uniform(-torch.pi, 2*torch.pi).rvs),
                         ('concentrated', vonmises(torch.pi/2, 2.).rvs)
                         ]
            }

    for loss_pair in losses:
        for rep in range(3):
            for encoding in ['linear']:
                for distribution_pair in dists[encoding]:
                    model = FaceAngle(loss_pair, distribution_pair, encoding)
                    trainModel(model,
                               f'trainedParameters/Exp2.2/{loss_pair[0]}_loss/{encoding}_enc/{distribution_pair[0]}/rep{rep}',
                               save_best_train=True
                               )
