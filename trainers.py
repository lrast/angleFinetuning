from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer

from basicModel import estimateAngle


def runBasicTraining(model):
    wandb_logger = WandbLogger(project='EstimateAngle')
    wandb_logger.experiment.config.update(model.hyperparameters )

    trainer = Trainer(logger=wandb_logger, max_epochs=model.hyperparameters['max_epochs'])
    trainer.fit(model)

