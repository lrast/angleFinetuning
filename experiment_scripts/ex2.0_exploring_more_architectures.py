from trainers import trainModel
from basicModel import EstimateAngle_Faces_experimental
from parameters import uniformConfig, concentratedConfig_1, \
                       concentratedConfig_2, lossFns

import torch
import glob
import numpy as np

from workingModels import FaceAngle
from scipy.stats import vonmises, norm, uniform
from scipy.stats import multinomial, dirichlet, rv_continuous


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


def Exp24_sweep_probs(N_samples):
    """
        The question here is whether adaptation to stimulus probability
        distributions depends on the whole distribution or simply on the

    """
    for N_bins in [4, 6]:
        probs = dirichlet(np.ones(N_bins)).rvs(N_samples)

        for i, p in enumerate(probs):
            torch.mps.empty_cache()
            distribution_pair = (f'random_{N_bins}', UniformMixture(p).rvs)

            model = FaceAngle(('linear', torch.nn.Identity()), distribution_pair, 'linear',
                              probs=p.tolist())
            trainModel(model,
                       f'trainedParameters/Exp2.4/{N_bins}bins/rep{i}',
                       save_best_train=True
                       )


class UniformMixture(rv_continuous):
    """UniformMixture: stepwise distribution over the domain -pi/2 to pi/2 """
    def __init__(self, probs):
        super(UniformMixture, self).__init__()
        self.probs = probs

        n_bins = len(probs)
        starts = np.linspace(-np.pi/2, np.pi/2, n_bins+1)[:-1]
        width = np.pi / n_bins

        self.dists = {i: uniform(starts[i], width) for i in range(n_bins)}

    def rvs(self, n_samples):
        samples = []
        num_per_bin = multinomial(n_samples, self.probs).rvs()[0]

        for i, n in enumerate(num_per_bin):
            samples.append(self.dists[i].rvs(n))

        samples = np.concatenate(samples)
        np.random.shuffle(samples)
        return samples


def Exp25_finetuning_test():
    """ 
        Checking out the behavior of these networks under fine-tuning
    """
    loss_pair = ('linear', torch.nn.Identity())
    dists = {
            'linear': [('uniform', uniform(-torch.pi/2, torch.pi).rvs),
                       ('concentrated', norm(torch.pi/4, 0.2).rvs)
                       ],
            'circular': [('uniform', uniform(-torch.pi, 2*torch.pi).rvs),
                         ('concentrated', vonmises(torch.pi/2, 2.).rvs)
                         ]
            }

    for rep in range(2):
        for encoding in ['linear', 'circular']:
            for i in range(2):
                distribution_pair = dists[encoding][i]

                directory = f'trainedParameters/Exp2.5/{encoding}_enc/{distribution_pair[0]}/rep{rep}/initial/'
                model = FaceAngle(loss_pair, distribution_pair, encoding)
                trainModel(model,
                           directory,
                           save_best_train=True
                           )

                ckpt = glob.glob(directory + 'train*')[0]
                distribution_pair_ft = dists[encoding][not i]

                model_ft = FaceAngle.load_from_checkpoint(ckpt,
                                                          distribution_pair=distribution_pair_ft,
                                                          dataSize=256)

                trainModel(model_ft,
                           f'trainedParameters/Exp2.5/{encoding}_enc/{distribution_pair_ft[0]}/rep{rep}/finetune_lowdata',
                           save_best_train=True
                           )

                model_ft_2 = FaceAngle.load_from_checkpoint(ckpt, distribution_pair=distribution_pair_ft)
                model_ft_2.model[0].requires_grad_(False)

                trainModel(model_ft_2,
                           f'trainedParameters/Exp2.5/{encoding}_enc/{distribution_pair[0]}/rep{rep}/finetune_frozen',
                           save_best_train=True
                           )
