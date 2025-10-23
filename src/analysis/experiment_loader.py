# tools for loading the results of experiments

import pandas as pd

from pathlib import Path
from omegaconf import OmegaConf
from tqdm import tqdm

from models.face_angle import EstimateAngle


def sweep_analysis(analysis, sweep_dir, out_path):
    """ Run a given analysis for every model in a sweep

        analysis: takes a function, returns a dictionary of results
    """
    root_directory = Path(sweep_dir)

    all_rows = []
    for i, experiment_dir in tqdm(enumerate(root_directory.iterdir())):
        if not experiment_dir.is_dir():
            continue

        model = EstimateAngle.load_from_checkpoint(experiment_dir / 'checkpoints/best.ckpt')
        config = OmegaConf.load(experiment_dir / '.hydra/config.yaml')

        result = analysis(model)
        row = {'kappa': config.data.kappa,
               'loss': config.model.loss_name}
        row.update(result)

        all_rows.append(row)

        pd.DataFrame(all_rows).to_parquet(out_path)
