# study of the effects of fine tuning on performance

import wandb
import glob

from basicModel import EstimateAngle
from trainers import runEarlyStoppingTraining

import torch
import pandas as pd


def Experiment5_retraining():
    """
        Does a pre-trained network suffer relative to a naive network under finetuning
    """

    dataSizes = [256, 1024]

    parameterSets = [(8., 0.), (8., torch.pi/2)]

    results = []

    for rep in range(3):
        for datasize in dataSizes:
            for i, parameters in enumerate(parameterSets):
                # train a network, run various finetunings, save all results

                # initial training 
                parentdir = get_model_directory(rep, datasize, parameters)
                loss = initialize_and_train(rep, datasize, parameters, parentdir)

                results_row = {'rep': rep, 'datasize': datasize, 'loc': parameters[1],
                               'kappa': parameters[0],
                               'rep_init': None, 'datasize_init': None, 'loc_init': None,
                               'kappa_init': None, 'loss': loss}

                results.append(results_row)

                saved_params = glob.glob(parentdir + '*')[0]

                # finetuning
                for rep_ft in range(3):
                    for datasize_ft in dataSizes:
                        parameters_ft = parameterSets[(i+1) % 2]

                        childdir = get_model_directory(rep, datasize, parameters,
                                                       rep_ft, datasize_ft, parameters_ft)

                        loss = initialize_and_train(rep_ft, datasize_ft,
                                                    parameters_ft, childdir)
                        results_row = {'rep': rep_ft, 'datasize': datasize_ft,
                                       'loc': parameters_ft[1], 'kappa': parameters_ft[0],
                                       'rep_init': rep, 'datasize_init': datasize,
                                       'loc_init': parameters[1], 'kappa_init': parameters[0],
                                       'loss': loss}

                        results.append(results_row)

    pd.DataFrame(results).to_csv('experiment_result/ex5_scan.csv', index=False)


def initialize_and_train(rep, datasize, parameters, directory, init_ckpt=None):
    # initial training on the data
    defaultConfig = {
                     'pixelDim': 101,
                     'shotNoise': 0.8,
                     'noiseVar': 20.,
                     }

    wandb.init(reinit=True)

    kappa, mu = parameters
    distribution_params = {'kappa_tr': kappa, 'kappa_test': kappa, 'kappa_val': kappa,
                           'loc_tr': mu, 'loc_val': mu, 'loc_test': mu,
                           'dataSize': datasize}

    if init_ckpt is None:
        model = EstimateAngle(**defaultConfig, **distribution_params, max_epochs=2500)
    else:
        model = EstimateAngle.load_from_checkpoint(init_ckpt, ** defaultConfig,
                                                   **distribution_params,
                                                   seed=torch.random.seed(),
                                                   max_epochs=2500)

    runEarlyStoppingTraining(model, directory, project='angleFineTuning')
    wandb.finish()

    # evaluate how well we did
    batch = next(iter(model.test_dataloader()))
    loss = model.test_step(batch, 0).item()
    return loss


def evaluate_models():
    """ The evaluations saved during training fail because they don't represent the best checkpoint """
    parameterSets = [(8., 0.), (8., torch.pi/2)]

    results = []

    for rep in range(3):
        for datasize in [256, 1024]:
            for i, parameters in enumerate(parameterSets):
                print(rep, datasize, parameters)
                parentdir = get_model_directory(rep, datasize, parameters)
                ckpt = glob.glob(parentdir + '*.ckpt')[0]
                model = EstimateAngle.load_from_checkpoint(ckpt)
                model.setup()
                batch = next(iter(model.test_dataloader()))
                batch['image'] = batch['image'].to(device=model.device)
                batch['angle'] = batch['angle'].to(device=model.device)

                loss = model.test_step(batch, 0).item()

                results_row = {'rep': rep, 'datasize': datasize, 'loc': parameters[1],
                               'kappa': parameters[0],
                               'rep_init': None, 'datasize_init': None, 'loc_init': None,
                               'kappa_init': None, 'loss': loss}

                results.append(results_row)

                for rep_ft in range(3):
                    for datasize_ft in [256, 1024]:
                        parameters_ft = parameterSets[(i+1) % 2]

                        childdir = get_model_directory(rep, datasize, parameters,
                                                       rep_ft, datasize_ft, parameters_ft)
                        ckpt = glob.glob(childdir + '*.ckpt')[0]
                        model = EstimateAngle.load_from_checkpoint(ckpt)
                        model.setup()
                        batch = next(iter(model.test_dataloader()))
                        batch['image'] = batch['image'].to(device=model.device)
                        batch['angle'] = batch['angle'].to(device=model.device)

                        loss = model.test_step(batch, 0).item()
                        
                        results_row = {'rep': rep_ft, 'datasize': datasize_ft,
                                       'loc': parameters_ft[1], 'kappa': parameters_ft[0],
                                       'rep_init': rep, 'datasize_init': datasize,
                                       'loc_init': parameters[1], 'kappa_init': parameters[0],
                                       'loss': loss}

                        results.append(results_row)

    pd.DataFrame(results).to_csv('experiment_result/ex5_scan_optimal.csv', index=False)


def get_model_directory(rep_par, datasize_par, parameters_par,
                        rep_child=None, datasize_child=None, parameters_child=None):
    parent_name = str(hash((rep_par, datasize_par, parameters_par)))
    child_name = str(hash((rep_child, datasize_child, parameters_child)))

    if rep_child is None:
        return f'trainedParameters/Exp5-retrain/{parent_name}/'
    else:
        return f'trainedParameters/Exp5-retrain/{parent_name}/{child_name}/'

