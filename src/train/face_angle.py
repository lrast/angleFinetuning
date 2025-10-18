# make trainer for face-angle experiments

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping


def make_trainer(dirpath, run_name, max_epochs=500, patience=50,
                 use_early_stopping=False, grad_clip=1.0,
                 project='EstimateAngle',
                 **ignored_kwargs
                 ):
    """ Make trainer for angle regression task """

    trainer_args = {
        'max_epochs': max_epochs,
        'accelerator': 'auto',
        'log_every_n_steps': 50,
        'gradient_clip_val': grad_clip,
    }

    logger = WandbLogger(project=project, name=run_name)

    checkpoint = ModelCheckpoint(dirpath=dirpath,
                                 filename='best',
                                 every_n_epochs=1, 
                                 save_top_k=1,
                                 monitor='eval/loss',
                                 save_weights_only=True,
                                 save_last=False
                                 )

    if use_early_stopping:
        earlystopping = EarlyStopping(monitor='eval/loss', mode='min', 
                                      patience=patience
                                      )
        callbacks = [checkpoint, earlystopping]

    else:
        callbacks = [checkpoint]

    trainer = Trainer(**trainer_args,
                      logger=logger,
                      callbacks=callbacks,
                      )

    return trainer
