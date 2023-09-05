from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from basicModel import EstimateAngle
import wandb


def runBasicTraining(model):
    """Simple training behavior with checkpointing"""
    wandb_logger = WandbLogger(project='EstimateAngle')
    wandb_logger.experiment.config.update(model.hyperparameters)

    checkpoint_callback = ModelCheckpoint(every_n_epochs=10, 
                                          save_top_k=4,
                                          monitor='Train Loss'
                                          )

    trainer = Trainer(logger=wandb_logger,
                      max_epochs=model.hyperparameters['max_epochs'],
                      callbacks=[checkpoint_callback]
                      )
    trainer.fit(model)


def RunSweep_TrainDistributionAndNoise():
    sweepCFG = {
        'method': 'grid',
        'name': 'training distribution',
        'parameters': {
            'kappa_tr': {'values': [0.1, 0.5, 1, 1.5]},
            'shotNoise': {'values': [0, 0.1, 0.5, 1.]},
            'noiseVar': {'values': [1.]}
        }
    }

    sweepid = wandb.sweep(sweep=sweepCFG, project='dist-vs-noise')
    wandb.agent(sweepid, lambda: sweepRun('training distribution'))


def sweepRun(projectname):
    """An indiviudual training run for a sweep"""
    wandb.init(project=projectname)
    model = EstimateAngle(**wandb.config)
    runBasicTraining(model)
