from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from basicModel import EstimateAngle
import wandb
import numpy as np


def runBasicTraining(model):
    """Simple training behavior with checkpointing"""
    wandb_logger = WandbLogger(project='EstimateAngle')
    wandb_logger.experiment.config.update(model.hparams)

    checkpoint_callback = ModelCheckpoint(every_n_epochs=10, 
                                          save_top_k=4,
                                          monitor='Train Loss'
                                          )

    trainer = Trainer(logger=wandb_logger,
                      max_epochs=model.hparams.max_epochs,
                      callbacks=[checkpoint_callback]
                      )
    trainer.fit(model)


def runEarlyStoppingTraining(model, directory, patience=200):
    """Simple training behavior with checkpointing"""
    wandb_logger = WandbLogger(project='EstimateAngle')

    earlystopping_callback = EarlyStopping(monitor='Val Loss', mode='min', 
                                           patience=patience
                                           )
    checkpoint_callback = ModelCheckpoint(dirpath=directory,
                                          every_n_epochs=1, 
                                          save_top_k=1,
                                          monitor='Val Loss'
                                          )

    trainer = Trainer(logger=wandb_logger,
                      max_epochs=model.hparams.max_epochs,
                      callbacks=[checkpoint_callback, earlystopping_callback]
                      )
    trainer.fit(model)
