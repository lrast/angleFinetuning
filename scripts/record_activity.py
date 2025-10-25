import torch
import sys
import numpy as np
import pandas as pd

from pathlib import Path
from omegaconf import OmegaConf
from tqdm import tqdm

from src.models.face_angle import EstimateAngle
from src.analysis.activity_discrimination import DiscriminationAnalysis

from sklearn.covariance import MinCovDet
from scipy.spatial.distance import mahalanobis


def sweep_activity_record(sweep_dir, name):
    """ Run a given analysis for every model in a sweep

        analysis: takes a function, returns a dictionary of results
    """
    root_directory = Path(sweep_dir)

    # hacky, but I'm recording the Fisher information measurements at the same time.
    all_rows = []
    for i, experiment_dir in tqdm(enumerate(root_directory.iterdir())):
        if not experiment_dir.is_dir():
            continue

        print(experiment_dir)
        model = EstimateAngle.load_from_checkpoint(experiment_dir / 'checkpoints/best.ckpt')
        model = model.to('mps')
        config = OmegaConf.load(experiment_dir / '.hydra/config.yaml')

        kappa = config.data.kappa
        loss = config.model.loss_name

        analyzer = DiscriminationAnalysis(model, model.decoder[2])

        FIs = []
        for i, midpoint in enumerate(np.linspace(-np.pi, np.pi, 50)):
            delta = 0.03

            location = Path(f'experiment_data/{name}/{kappa}/{loss}/{i}/')
            location.mkdir(parents=True)

            data, labels = analyzer.get_activity(midpoint, delta)
            torch.save(data, location / 'data.pt')
            torch.save(labels, location / 'labels.pt')

        row = {'kappa': kappa,
               'loss': loss,
               'fisher_info': np.array(FIs)}

        all_rows.append(row)

        pd.DataFrame(all_rows).to_parquet('experiment_data/fi_robust.parquet')


def robust_FI(data, labels, delta):
    mcd0 = MinCovDet().fit(data[labels == 0])
    mu_robust0 = mcd0.location_
    cov_robust0 = mcd0.covariance_
    
    mcd1 = MinCovDet().fit(data[labels == 1])
    mu_robust1 = mcd1.location_
    cov_robust1 = mcd1.covariance_

    return mahalanobis(mu_robust0, mu_robust1,
                       np.linalg.inv(0.5*cov_robust1 + 0.5*cov_robust0)
                       ) / delta


if __name__ == '__main__':
    print(sys.argv[1])
    sweep_activity_record(sys.argv[1], sys.argv[2])
