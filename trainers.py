import torch
import wandb

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping


def runBasicTraining(model):
    """Simple training behavior with checkpointing"""
    wandb_logger = WandbLogger(project='EstimateAngle')
    wandb_logger.experiment.config.update(model.hparams)

    checkpoint_callback = ModelCheckpoint(every_n_epochs=10, 
                                          save_top_k=4,
                                          monitor='Train Loss',
                                          save_weights_only=True
                                          )

    trainer = Trainer(logger=wandb_logger,
                      max_epochs=model.hparams.max_epochs,
                      callbacks=[checkpoint_callback]
                      )
    trainer.fit(model)


def runEarlyStoppingTraining(model, directory, project='EstimateAngle', patience=200, **trainer_kwargs):
    """Simple training behavior with checkpointing"""
    wandb.init(reinit=True)

    wandb_logger = WandbLogger(project=project)

    earlystopping_callback = EarlyStopping(monitor='Val Loss', mode='min', 
                                           patience=patience
                                           )
    checkpoint_callback = ModelCheckpoint(dirpath=directory,
                                          every_n_epochs=1, 
                                          save_top_k=1,
                                          monitor='Val Loss',
                                          save_weights_only=True,
                                          save_last=False
                                          )

    trainer = Trainer(logger=wandb_logger,
                      max_epochs=model.hparams.max_epochs,
                      callbacks=[checkpoint_callback, earlystopping_callback],
                      **trainer_kwargs
                      )
    trainer.fit(model)

    wandb.finish()

    return trainer.checkpoint_callback.best_model_path


def trainEarlyStoppingAndLoad(model, directory, project='EstimateAngle', patience=200):
    """Train with checkpointing, and load the best weights into the model"""
    wandb.init(reinit=True)

    wandb_logger = WandbLogger(project=project)

    earlystopping_callback = EarlyStopping(monitor='Val Loss', mode='min', 
                                           patience=patience
                                           )
    checkpoint_callback = ModelCheckpoint(dirpath=directory,
                                          every_n_epochs=1, 
                                          save_top_k=1,
                                          monitor='Val Loss',
                                          save_weights_only=True
                                          )

    trainer = Trainer(logger=wandb_logger,
                      max_epochs=model.hparams.max_epochs,
                      callbacks=[checkpoint_callback, earlystopping_callback]
                      )
    trainer.fit(model)

    wandb.finish()

    best_model_data = torch.load(trainer.checkpoint_callback.best_model_path,
                                 map_location=model.device)
    model.load_state_dict(best_model_data['state_dict'])


def trainEarlyStoppingAndLoad_customTest(model, directory, project='EstimateAngle',
                                         patience=200, save_weights_only=True):
    """Train with checkpointing, and load the best weights into the model
    Able to save only the weights or the trainer state as well.
    """
    wandb.init(reinit=True)

    wandb_logger = WandbLogger(project=project)

    earlystopping_callback = EarlyStopping(monitor='Val Loss', mode='min', 
                                           patience=patience
                                           )
    checkpoint_callback = ModelCheckpoint(dirpath=directory,
                                          every_n_epochs=1, 
                                          save_top_k=1,
                                          monitor='Val Loss',
                                          save_weights_only=save_weights_only
                                          )

    trainer = Trainer(logger=wandb_logger,
                      max_epochs=model.hparams.max_epochs,
                      callbacks=[checkpoint_callback, earlystopping_callback]
                      )
    trainer.fit(model)

    wandb.finish()

    ckpt_path = trainer.checkpoint_callback.best_model_path
    best_model_data = torch.load(ckpt_path,
                                 map_location=model.device)
    model.load_state_dict(best_model_data['state_dict'])

    return ckpt_path


def trainModel(model, directory, project='EstimateAngle', patience=200, **trainer_kwargs):
    """Simple training behavior with checkpointing"""
    wandb.init(reinit=True)

    trainer_params = ['max_epochs', 'gradient_clip_val']
    model_defaults = dict(filter(lambda i: (i[0] in trainer_params), model.hparams.items()))
    model_defaults.update(trainer_kwargs)
    trainer_kwargs = model_defaults

    wandb_logger = WandbLogger(project=project)

    earlystopping_callback = EarlyStopping(monitor='Val Loss', mode='min', 
                                           patience=patience
                                           )

    checkpoint_callback = ModelCheckpoint(dirpath=directory,
                                          every_n_epochs=1, 
                                          save_top_k=1,
                                          monitor='Val Loss',
                                          save_weights_only=True,
                                          save_last=False
                                          )

    trainer = Trainer(logger=wandb_logger,
                      callbacks=[checkpoint_callback, earlystopping_callback],
                      **trainer_kwargs
                      )
    trainer.fit(model)

    wandb.finish()

    return trainer.checkpoint_callback.best_model_path
