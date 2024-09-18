
import wandb
import glob

import numpy as np

from basicModel import EstimateAngle_Faces
from trainers import runEarlyStoppingTraining

import torch
from torch import nn
from torch.nn import CosineSimilarity


def Experiment8_distribution():
    """ Training models on the face dataset with different stimulus distributions:
        Can the model learn to be more accurate than uniform when the data is very concentrated?
    """
    concentratedConfig = {
                 'loc_tr': np.pi/2,
                 'kappa_tr': 8.,
                 'loc_val': np.pi/2,
                 'kappa_val': 8.,
                 'loc_test': np.pi/2,
                 'kappa_test': 8.
                 }

    uniformConfig = {
                 'loc_tr': np.pi/2,
                 'kappa_tr': 1E-8,
                 'loc_val': np.pi/2,
                 'kappa_val': 1E-8,
                 'loc_test': np.pi/2,
                  'kappa_test': 1E-8
                 }


    # Cosine similarity is large (close to one) if the two vectors are
    # similar to each other
    cosSim = CosineSimilarity()
    eps = 1E-16

    for rep in range(4):
        conc_model = EstimateAngle_Faces(**concentratedConfig,
                              max_epochs=4000
                              )

        runEarlyStoppingTraining(conc_model,
                                 directory=f'trainedParameters/Exp8/initial_check/concentrated/rep{rep}/')

        unif_model = EstimateAngle_Faces(**uniformConfig,
                              max_epochs=4000
                              )

        runEarlyStoppingTraining(unif_model,
                                 directory=f'trainedParameters/Exp8/initial_check/uniform/rep{rep}/')


def Experiment8_loss_function_effect():
    """ Training models with different loss functions wrapping the cosine similarity:
        Do differences in the outputs emerge?

        We are now applying this to the face dataset, where differences in coding fidelity can emerge
    """
    uniformConfig = {
                 'loc_tr': np.pi/2,
                 'kappa_tr': 1E-8,
                 'loc_val': np.pi/2,
                 'kappa_val': 1E-8,
                 'loc_test': np.pi/2,
                  'kappa_test': 1E-8
                 }

    concentratedConfig_1 = {
             'loc_tr': np.pi/2,
             'kappa_tr': 3.,
             'loc_val': np.pi/2,
             'kappa_val': 3.,
             'loc_test': np.pi/2,
             'kappa_test': 3.
             }

    concentratedConfig_2 = {
             'loc_tr': np.pi/2,
             'kappa_tr': 1.,
             'loc_val': np.pi/2,
             'kappa_val': 1.,
             'loc_test': np.pi/2,
             'kappa_test': 1.
             }


    # Cosine similarity is large (close to one) if the two vectors are
    # similar to each other
    cosSim = CosineSimilarity()
    eps = 1E-16


    lossFns = {
        'linear': lambda x, y: 1.-torch.mean(cosSim(x, y)),
        'log': lambda x, y: np.log(2.)-torch.mean(torch.log(cosSim(x, y) + 1. + eps)),
        'sqrt': lambda x, y: np.sqrt(2)-torch.mean(torch.sqrt(cosSim(x, y) + 1. + eps))
    }

    distParams = {
        'uniform': uniformConfig,
        'high_conc': concentratedConfig_1,
        'low_conc': concentratedConfig_2
    }

    for rep in range(4):
        for loss_name in lossFns.keys():
            for dist_name in distParams.keys():
                model = EstimateAngle_Faces(**distParams[dist_name],
                              max_epochs=4000
                              )
                model.lossFn = lossFns[loss_name]

                runEarlyStoppingTraining(model,
                    directory=f'trainedParameters/Exp8/loss/{loss_name}/{dist_name}/rep{rep}/')


def loss_fn_debug(hypothesis=1):
    uniformConfig = {
             'loc_tr': np.pi/2,
             'kappa_tr': 1E-8,
             'loc_val': np.pi/2,
             'kappa_val': 1E-8,
             'loc_test': np.pi/2,
              'kappa_test': 1E-8
             }

    cosSim = CosineSimilarity()
    eps = 1E-16
    relu = torch.nn.ReLU()

    positive_sim = lambda x, y: relu(cosSim(x,y) + 1) + eps

    lossFns = {
        'log': lambda x, y: np.log(2.)-torch.mean(torch.log(positive_sim(x,y))),
        'sqrt': lambda x, y: np.sqrt(2)-torch.mean(torch.sqrt(positive_sim(x,y)))
    }


    """ The non-linear loss functions didn't learn uniform distributions: investigating """
    if hypothesis == 1:
        # the epsilon is not enought to keep the loss from going negative
        for rep in range(4):
            for loss_name in lossFns.keys():
                model = EstimateAngle_Faces(**uniformConfig,
                              max_epochs=4000,
                              loss_fn = loss_name
                              )
                model.lossFn = lossFns[loss_name]

                runEarlyStoppingTraining(model,
                    directory=f'trainedParameters/Exp8/debug_1/{loss_name}/rep{rep}/')


def loss_fn_debug(hypothesis=1):
    uniformConfig = {
             'loc_tr': np.pi/2,
             'kappa_tr': 1E-8,
             'loc_val': np.pi/2,
             'kappa_val': 1E-8,
             'loc_test': np.pi/2,
              'kappa_test': 1E-8
             }

    cosSim = CosineSimilarity()
    eps = 1E-16
    relu = torch.nn.ReLU()

    positive_sim = lambda x, y: relu(cosSim(x,y) + 1) + eps

    lossFns = {
        'log': lambda x, y: np.log(2.)-torch.mean(torch.log(positive_sim(x,y))),
        'sqrt': lambda x, y: np.sqrt(2)-torch.mean(torch.sqrt(positive_sim(x,y)))
    }


    """ The non-linear loss functions didn't learn uniform distributions: investigating """
    if hypothesis == 1:
        # the epsilon is not enought to keep the loss from going negative
        for rep in range(4):
            for loss_name in lossFns.keys():
                model = EstimateAngle_Faces(**uniformConfig,
                              max_epochs=4000,
                              loss_fn = loss_name
                              )
                model.lossFn = lossFns[loss_name]

                runEarlyStoppingTraining(model,
                    directory=f'trainedParameters/Exp8/debug_1/{loss_name}/rep{rep}/')


def Experiment8_loss_function_effect_gradient_clip():
    """ Exploring face orientation learning: with neural network improvements
    """
    uniformConfig = {
                 'loc_tr': np.pi/2,
                 'kappa_tr': 1E-8,
                 'loc_val': np.pi/2,
                 'kappa_val': 1E-8,
                 'loc_test': np.pi/2,
                  'kappa_test': 1E-8
                 }

    concentratedConfig_1 = {
             'loc_tr': np.pi/2,
             'kappa_tr': 3.,
             'loc_val': np.pi/2,
             'kappa_val': 3.,
             'loc_test': np.pi/2,
             'kappa_test': 3.
             }

    concentratedConfig_2 = {
             'loc_tr': np.pi/2,
             'kappa_tr': 1.,
             'loc_val': np.pi/2,
             'kappa_val': 1.,
             'loc_test': np.pi/2,
             'kappa_test': 1.
             }


    # Cosine similarity is large (close to one) if the two vectors are
    # similar to each other
    cosSim = CosineSimilarity()
    eps = 1E-8
    relu = torch.nn.ReLU()

    positive_sim = lambda x, y: relu(cosSim(x,y) + 1 + eps/2) + eps/2

    lossFns = {
        'linear': lambda x, y: 2. + eps - torch.mean(positive_sim(x, y)),
        'log': lambda x, y: np.log(2. + eps) - torch.mean(torch.log(positive_sim(x, y))),
        'sqrt': lambda x, y: np.sqrt(2 + eps) - torch.mean(torch.sqrt(positive_sim(x, y)))
    }

    distParams = {
        'uniform': uniformConfig,
        'high_conc': concentratedConfig_1,
        'low_conc': concentratedConfig_2
    }

    for rep in range(5):
        for loss_name in lossFns.keys():
            for dist_name in distParams.keys():
                model = EstimateAngle_Faces(**distParams[dist_name],
                              max_epochs=4000,
                              loss_fn = loss_name
                              )
                model.lossFn = lossFns[loss_name]

                runEarlyStoppingTraining(model,
                    directory=f'trainedParameters/Exp8/loss3/{loss_name}/{dist_name}/rep{rep}/',
                    gradient_clip_val=0.5
                    )

def Experiment82_smaller_networks(hidden_dims):
    """ Differences do emerge between different loss functions and different stimulus distributions.
        However, the difference in Mean square error is pretty slight.
        Here, we are trying to enhance the differences by using smaller networks, effectively making
        the task more difficult.
    """
    uniformConfig = {
                 'loc_tr': np.pi/2,
                 'kappa_tr': 1E-8,
                 'loc_val': np.pi/2,
                 'kappa_val': 1E-8,
                 'loc_test': np.pi/2,
                  'kappa_test': 1E-8
                 }

    concentratedConfig_1 = {
             'loc_tr': np.pi/2,
             'kappa_tr': 3.,
             'loc_val': np.pi/2,
             'kappa_val': 3.,
             'loc_test': np.pi/2,
             'kappa_test': 3.
             }

    concentratedConfig_2 = {
             'loc_tr': np.pi/2,
             'kappa_tr': 1.,
             'loc_val': np.pi/2,
             'kappa_val': 1.,
             'loc_test': np.pi/2,
             'kappa_test': 1.
             }


    # Cosine similarity is large (close to one) if the two vectors are
    # similar to each other
    cosSim = CosineSimilarity()
    eps = 1E-8
    relu = torch.nn.ReLU()

    positive_sim = lambda x, y: relu(cosSim(x,y) + 1 + eps/2) + eps/2

    lossFns = {
        'linear': lambda x, y: 2. + eps - torch.mean(positive_sim(x, y)),
        'log': lambda x, y: np.log(2. + eps) - torch.mean(torch.log(positive_sim(x, y))),
        'sqrt': lambda x, y: np.sqrt(2 + eps) - torch.mean(torch.sqrt(positive_sim(x, y)))
    }

    distParams = {
        'uniform': uniformConfig,
        'high_conc': concentratedConfig_1,
        'low_conc': concentratedConfig_2
    }

    for rep in range(3):
        for loss_name in lossFns.keys():
            for dist_name in distParams.keys():
                model = EstimateAngle_Faces(**distParams[dist_name],
                              max_epochs=4000,
                              loss_fn = loss_name,
                              hidden_dims = hidden_dims
                              )
                model.lossFn = lossFns[loss_name]

                runEarlyStoppingTraining(model,
                    directory=f'trainedParameters/Exp8/smaller/{loss_name}/{dist_name}/rep{rep}/',
                    gradient_clip_val=0.5
                    )


def Experiment83_with_dropout():
    """ Differences do emerge between different loss functions and different stimulus distributions.
        However, the difference in Mean square error is pretty slight.
        Here, we are trying to enhance the differences by using smaller networks, effectively making
        the task more difficult.
    """
    uniformConfig = {
                 'loc_tr': np.pi/2,
                 'kappa_tr': 1E-8,
                 'loc_val': np.pi/2,
                 'kappa_val': 1E-8,
                 'loc_test': np.pi/2,
                  'kappa_test': 1E-8
                 }

    concentratedConfig_1 = {
             'loc_tr': np.pi/2,
             'kappa_tr': 3.,
             'loc_val': np.pi/2,
             'kappa_val': 3.,
             'loc_test': np.pi/2,
             'kappa_test': 3.
             }

    concentratedConfig_2 = {
             'loc_tr': np.pi/2,
             'kappa_tr': 1.,
             'loc_val': np.pi/2,
             'kappa_val': 1.,
             'loc_test': np.pi/2,
             'kappa_test': 1.
             }


    # Cosine similarity is large (close to one) if the two vectors are
    # similar to each other
    cosSim = CosineSimilarity()
    eps = 1E-8
    relu = torch.nn.ReLU()

    positive_sim = lambda x, y: relu(cosSim(x,y) + 1 + eps/2) + eps/2

    lossFns = {
        'linear': lambda x, y: 2. + eps - torch.mean(positive_sim(x, y)),
        'log': lambda x, y: np.log(2. + eps) - torch.mean(torch.log(positive_sim(x, y))),
        'sqrt': lambda x, y: np.sqrt(2 + eps) - torch.mean(torch.sqrt(positive_sim(x, y)))
    }

    distParams = {
        'uniform': uniformConfig,
        'high_conc': concentratedConfig_1,
        'low_conc': concentratedConfig_2
    }

    hidden_dims = (30, 10)
    pixelDim = 64

    for rep in range(3):
        for loss_name in lossFns.keys():
            for dist_name in distParams.keys():
                model = EstimateAngle_Faces(**distParams[dist_name],
                              max_epochs=4000,
                              loss_fn = loss_name,
                              hidden_dims = hidden_dims
                              )
                model.model = nn.Sequential(
                    nn.Linear(pixelDim**2, hidden_dims[0]),
                    nn.LeakyReLU(),
                    nn.Dropout(),
                    nn.Linear(hidden_dims[0], hidden_dims[1]),
                    nn.LeakyReLU(),
                    nn.Dropout(),
                    nn.Linear(hidden_dims[1], 2)
                )
                model.lossFn = lossFns[loss_name]

                runEarlyStoppingTraining(model,
                    directory=f'trainedParameters/Exp8/dropout/{loss_name}/{dist_name}/rep{rep}/',
                    gradient_clip_val=0.5
                    )
