import wandb
import glob
import numpy as np

from basicModel import EstimateAngle
from trainers import runEarlyStoppingTraining

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

#  Experinment 1: sweeping across different noise levels and variances in the
#  training data.
#  Goal: find accuracy that depends on training mean


def InitialSweep_TrainDistributionAndNoise():
    sweepCFG = {
        'method': 'grid',
        'name': 'DistAndNoiseCharacterization',
        'parameters': {
            'reps': {'values': [0, 1]},
            'kappa_tr': {'values': [1., 2., 4.]},
            'loc_tr': {'values': (np.pi * np.linspace(0, 2, 7)).tolist()[0:2]},
            'shotNoise': {'values': [0.8]},
            'noiseVar': {'values': [1., 10., 20.]},
            'pixelDim': {'values': [101]}
        }
    }

    def sweepRun():
        """An individual training run for a sweep"""
        wandb.init()

        print(dict(wandb.config).values)

        dirName = str(hash(tuple(dict(wandb.config).values())))[1:]

        model = EstimateAngle(**wandb.config,
                              loc_val=wandb.config['loc_tr'],
                              kappa_val=wandb.config['kappa_tr'],
                              max_epochs=3000
                              )
        runEarlyStoppingTraining(model, 
                                 directory=f'initialSweepResults/{dirName}')

    sweepid = wandb.sweep(sweep=sweepCFG, project='EstimateAngle')
    wandb.agent(sweepid, sweepRun)


def dataFile_initialSweep(kappa_tr, loc_tr, noiseVar, 
                          pixelDim, reps, shotNoise):
    dataDirectory = 'initialSweepResults/' + str(hash((kappa_tr, loc_tr,
                                                 noiseVar, pixelDim, reps,
                                                 shotNoise)))[1:]

    return glob.glob(dataDirectory+'/*')[0]


#  Experinment 2: A bit of a re-run of experiment 1, with a bug fix
#  knowing the right noise level
#  Goal: sensitivity scaling with 


def Experiment2_trainingSetSize():
    """ What is the impact of training on distributions concentrated
        on different means?
        How is this modified by  
    """

    defaultConfig = {
                     'pixelDim': 101,
                     'shotNoise': 0.8,
                     'noiseVar': 20.
                     }

    concentratedSweep = {
        'method': 'grid',
        'name': 'Exp2concentrated',
        'parameters': {
            'reps': {'values': [0, 1]},
            'loc_tr': {'values': [0., np.pi/3]},
            'kappa_tr': {'values': [4.]},
            'dataSize': {'values': [512, 1024, 2048]}
        }
    }

    flatSweep = {
        'method': 'grid',
        'name': 'Exp2flat',
        'parameters': {
            'reps': {'values': [0, 1]},
            'loc_tr': {'values': [0.]},
            'kappa_tr': {'values': [1E-16]},
            'dataSize': {'values': [512, 1024, 2048]}
        }
    }

    def sweepRun():
        """An individual training run for a sweep"""
        wandb.init()

        print(dict(wandb.config))

        dirName = str(hash(tuple(dict(wandb.config).values())))[1:]

        model = EstimateAngle(**defaultConfig,
                              **wandb.config,
                              loc_val=wandb.config['loc_tr'],
                              kappa_val=wandb.config['kappa_tr'],
                              max_epochs=3000
                              )
        runEarlyStoppingTraining(model, 
                                 directory=f'Experiment2/{dirName}')

    ConcentratedSweepID = wandb.sweep(sweep=concentratedSweep,
                                      project='EstimateAngle')
    wandb.agent(ConcentratedSweepID, sweepRun)
    FlatSweepID = wandb.sweep(sweep=flatSweep, project='EstimateAngle')
    wandb.agent(FlatSweepID, sweepRun)


def dataFile_Experiment2(dataSize, kappa_tr, loc_tr, reps):
    dataDirectory = 'Experiment2/' + str(hash((dataSize, kappa_tr, 
                                              loc_tr, reps)))[1:]

    return glob.glob(dataDirectory+'/*')[0]


def Experiment3_fineTuning():
    """ Objective: fine tune each network on a different distribution of angles
        see how the sensitivity curve changes
     """

    modelParams = {
        0: [0., 4.],
        1: [np.pi/3, 4.],
        2: [0., 1E-16]
    }

    dataSizes = [50, 100, 200]

    def fineTune(model, directory):
        max_epochs = 400
        every_n_epochs = 20

        wandb.init(project='EstimateAngle')
        wandb_logger = WandbLogger(project='EstimateAngle')

        earlystopping_callback = EarlyStopping(monitor='Val Loss', mode='min', 
                                               patience=50
                                               )
        checkpoint_callback = ModelCheckpoint(dirpath=directory,
                                              every_n_epochs=every_n_epochs, 
                                              save_top_k=-1,
                                              monitor='Val Loss'
                                              )

        trainer = Trainer(logger=wandb_logger,
                          max_epochs=max_epochs,
                          callbacks=[checkpoint_callback, earlystopping_callback]
                          )
        trainer.fit(model)
        wandb.finish()

    for fineTuneSize in dataSizes:
        for ind_model in range(3):
            for rep in range(2):
                loc_tr, kappa_tr = modelParams[ind_model]

                model = EstimateAngle.load_from_checkpoint(
                                                dataFile_Experiment2(
                                                    512, kappa_tr, loc_tr, rep
                                                ))

                finetuneIndex = (ind_model + 1) % 2
                loc_ft, kappa_ft = modelParams[finetuneIndex]
                model.hparams.loc_tr = loc_ft
                model.hparams.loc_val = loc_ft
                model.hparams.kappa_tr = kappa_ft
                model.hparams.kappa_val = kappa_ft
                model.hparams.dataSize = fineTuneSize

                directory = 'finetune/' + str(hash(
                                        (fineTuneSize, kappa_tr, loc_tr, rep)
                                                    ))[1:]

                fineTune(model, directory)


def dataFile_fineTune1(fineTuneSize, kappa_tr, loc_tr, rep):
    directory = 'finetune/' + str(hash(
                                    (fineTuneSize, kappa_tr, loc_tr, rep)
                                    ))[1:]
    return glob.glob(directory+'/*')







