import glob

import numpy as np

from basicModel import EstimateAngle_Faces_final
from trainers import trainModel

import torch
from torch import nn
from torch.nn import CosineSimilarity


def Experiment9_finetuning():
    """
    Running the fine-tuning with new angle distributions, but the same training data

    For now, I'm retraining all replicates once each and not worrying about the seeds.

    One idea to note: it might be worth using another split of the dataset for the finetuning experiments 
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

    lossFns = {
        'linear': lambda d: 2. - torch.mean(d),
        'log': lambda d: np.log(2.) - torch.mean(torch.log(d)),
        'sqrt': lambda d: np.sqrt(2.) - torch.mean(torch.sqrt(d))
    }

    distParams = {
        'uniform': uniformConfig,
        'high_conc': concentratedConfig_1,
        'low_conc': concentratedConfig_2
    }

    hidden_dims = (30, 10)
    pixelDim = 64


    for loss_pair in lossFns.items():
        loss_name = loss_pair[0]
        for old_dist in distParams.keys():
            for old_rep in range(3):
                old_ckpt = glob.glob(f'trainedParameters/Exp8/dropout/{loss_name}/{old_dist}/rep{old_rep}/*')[0]

                for new_dist in distParams.keys():
                    if new_dist == old_dist:
                        continue

                    new_ckpt_dir = f'trainedParameters/Exp9/finetune/{loss_name}/{old_dist}_to_{new_dist}/rep{old_rep}/'
                    retrain_params = distParams[new_dist]

                    model = EstimateAngle_Faces_final.load_from_checkpoint(old_ckpt, loss_pair=loss_pair, **retrain_params)

                    trainModel(model, directory=new_ckpt_dir)

                    del model


def Experiment9_data_size_sweep():
    """
    Sweeping through the different data set sizes
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

    loss_pair = ('linear', lambda d: 2. - torch.mean(d))

    distParams = {
        'uniform': uniformConfig,
        'high_conc': concentratedConfig_1,
        'low_conc': concentratedConfig_2
    }

    hidden_dims = (30, 10)
    pixelDim = 64

    datasizes = [512, 256, 128, 64, 32, 16]

    for datasize in datasizes:
        loss_name = loss_pair[0]
        for old_dist in distParams.keys():
            for old_rep in range(3):
                old_ckpt = glob.glob(f'trainedParameters/Exp8/dropout/{loss_name}/{old_dist}/rep{old_rep}/*')[0]

                for new_dist in distParams.keys():
                    if new_dist == old_dist:
                        continue

                    new_ckpt_dir = f'trainedParameters/Exp9/size_sweep/size{datasize}/{old_dist}_to_{new_dist}/rep{old_rep}/'
                    retrain_params = distParams[new_dist]

                    model = EstimateAngle_Faces_final.load_from_checkpoint(old_ckpt, loss_pair=loss_pair,
                                                                           dataSize=datasize,
                                                                           **retrain_params)

                    trainModel(model, directory=new_ckpt_dir)

                    del model
                    torch.cuda.empty_cache()


def Experiment9_frozen_layers():
    """
    Sweeping through the different data set sizes
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

    loss_pair = ('linear', lambda d: 2. - torch.mean(d))

    distParams = {
        'uniform': uniformConfig,
        'high_conc': concentratedConfig_1,
        'low_conc': concentratedConfig_2
    }

    hidden_dims = (30, 10)
    pixelDim = 64

    n_layers_to_freeze = [1, 2]

    for frozen in n_layers_to_freeze:
        for old_dist in distParams.keys():
            for old_rep in range(3):
                old_ckpt = glob.glob(f'trainedParameters/Exp8/dropout/{loss_pair[0]}/{old_dist}/rep{old_rep}/*')[0]

                for new_dist in distParams.keys():
                    if new_dist == old_dist:
                        continue

                    new_ckpt_dir = f'trainedParameters/Exp9/freeze_layers/{frozen}layers/{old_dist}_to_{new_dist}/rep{old_rep}/'
                    retrain_params = distParams[new_dist]

                    model = EstimateAngle_Faces_final.load_from_checkpoint(old_ckpt, loss_pair=loss_pair,
                                                                           **retrain_params)

                    # freeze the layers
                    for i, param in enumerate(model.parameters()):
                        if i < 2*frozen:
                            param.requires_grad = False

                    trainModel(model, directory=new_ckpt_dir)

                    del model
                    torch.cuda.empty_cache()
