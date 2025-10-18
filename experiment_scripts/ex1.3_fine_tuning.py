# Experiment 3: fine tuning the networks

import wandb
import glob
import numpy as np

from basicModel import EstimateAngle

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from ex2_sensitivity_scaling import dataFile_Experiment2


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

                directory = 'trainedParameters/finetune/' + str(hash(
                                        (fineTuneSize, kappa_tr, loc_tr, rep)
                                                    ))[1:]

                fineTune(model, directory)


def dataFile_fineTune1(fineTuneSize, kappa_tr, loc_tr, rep):
    directory = 'trainedParameters/finetune/' + str(hash(
                                                (fineTuneSize, kappa_tr, loc_tr, rep)
                                                ))[1:]
    return glob.glob(directory+'/*')
