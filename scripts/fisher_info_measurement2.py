import sys
import numpy as np
import pandas as pd

from pathlib import Path
from omegaconf import OmegaConf
from tqdm import tqdm

from src.models.face_angle import EstimateAngle
from src.analysis.activity_discrimination import DiscriminationAnalysis


def sweep_fisher_info(sweep_dir, name):
    """ Run a given analysis for every model in a sweep
    """
    root_directory = Path(sweep_dir)

    all_rows = []
    for i, experiment_dir in tqdm(enumerate(root_directory.iterdir())):
        if not experiment_dir.is_dir():
            continue

        print(experiment_dir)
        model = EstimateAngle.load_from_checkpoint(experiment_dir / 'checkpoints/best.ckpt')
        model = model.to('mps')
        config = OmegaConf.load(experiment_dir / '.hydra/config.yaml')

        kappa = config.data.kappa
        lagrange = config.model.lagrange

        analyzer = DiscriminationAnalysis(model, model.decoder[2])

        FIs = analyzer.Fisher_info(np.linspace(-np.pi, np.pi, 200), delta=0.03)

        row = {'kappa': kappa,
               'loss': lagrange,
               'fisher_info': np.array(FIs)}

        all_rows.append(row)

        pd.DataFrame(all_rows).to_parquet(f'experiment_data/{name}.parquet')


if __name__ == '__main__':
    print(sys.argv[1])
    sweep_fisher_info(sys.argv[1], sys.argv[2])
