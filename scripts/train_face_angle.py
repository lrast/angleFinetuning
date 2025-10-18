# Train script for face angle models
import hydra

from torch.utils.data import DataLoader
from omegaconf import DictConfig,  OmegaConf
from hydra.core.hydra_config import HydraConfig

from src.data.rotated_faces import RotatedFacesDataset, SingleFaceDataset
from src.models.face_angle import EstimateAngle
from src.train.face_angle import make_trainer


@hydra.main(config_path="../config", config_name="base", version_base="1.2")
def main(cfg: DictConfig):
    # Initialize data
    if cfg.debug.single_image:
        data_train = SingleFaceDataset(**cfg.data, split='train')
        data_val = SingleFaceDataset(**cfg.data, split='valid')
    else:
        data_train = RotatedFacesDataset(**cfg.data, split='train')
        data_val = RotatedFacesDataset(**cfg.data, split='valid')

    train_dl = DataLoader(data_train,
                          batch_size=cfg.train.batch_size,
                          shuffle=True,
                          num_workers=4,
                          persistent_workers=True)
    val_dl = DataLoader(data_val,
                        batch_size=cfg.train.batch_size,
                        num_workers=4,
                        persistent_workers=True)

    # initialize model
    if cfg.model.base_model.split('/')[0] == 'lrast':
        model_cfg = OmegaConf.to_container(cfg.model, resolve=True) 
        base_name = model_cfg.pop('base_model')
        model = EstimateAngle.from_pretrained(base_name, **model_cfg, lr=cfg.train.lr)
    else:
        model = EstimateAngle(**cfg.model, lr=cfg.train.lr)

    if cfg.train.frozen_base:
        model.freeze_base()

    model.train()

    # Initialize trainer
    trainer = make_trainer(dirpath=HydraConfig.get().runtime.output_dir+'/checkpoints/',
                           **cfg.train)

    # Train
    trainer.fit(model, train_dl, val_dl)


if __name__ == "__main__":
    main()
