# Linear discriminator on activity data
import torch
import numpy as np
import warnings

from ..data.rotated_faces import ConsistentRotationDataset

from sklearn.model_selection import StratifiedKFold
from sklearn.covariance import MinCovDet
from scipy.spatial.distance import mahalanobis


from tqdm import tqdm


class ActivityRecord():
    """Recorder to capture internal model activity"""
    def __init__(self, module):
        self.activity = {}
        self.hooks = module.register_forward_hook(self.record_activity)

        self.activity = {}
        self.key = 'none'

    def set_key(self, key):
        self.key = key

    def record_activity(self, module, input, output):
        key = self.key
        output = output.detach().clone()

        if key not in self.activity:
            self.activity[key] = output
        else: 
            self.activity[key] = torch.concat([self.activity[key], output])

    def clear_record(self):
        self.activity = {}


class DiscriminationAnalysis():
    """DiscriminationAnalysis: holder for the discrimination based Fisher information 
    analyses

    To do: Update this to use robust statistic for FI measurements
    """
    def __init__(self, model, module):
        self.model = model
        self.module = module

        self.recorder = ActivityRecord(module)
        self.N_samples = 19867

    def get_activity(self, midpoint, delta):
        """ performance of linear discriminator on internal activity data"""
        ds1 = ConsistentRotationDataset(midpoint-delta/2, split='valid')
        ds2 = ConsistentRotationDataset(midpoint+delta/2, split='valid')

        N_samples = self.N_samples
        if N_samples < len(ds1):
            inds1 = torch.randperm(10000)[0:N_samples]
            inds2 = torch.randperm(10000)[0:N_samples]

            ds1 = torch.utils.data.Subset(ds1, inds1)
            ds2 = torch.utils.data.Subset(ds2, inds2)

        dl1 = torch.utils.data.DataLoader(ds1, batch_size=32,
                                          shuffle=False, num_workers=4,
                                          )
        dl2 = torch.utils.data.DataLoader(ds2, batch_size=32,
                                          shuffle=False, num_workers=4,
                                          )

        self.recorder.set_key(0)
        for batch in iter(dl1):
            self.model.forward(batch[0].to(self.model.device))

        self.recorder.set_key(1)
        for batch in iter(dl2):
            self.model.forward(batch[0].to(self.model.device))

        l1 = len(self.recorder.activity[0])
        l2 = len(self.recorder.activity[1])
        all_embeddings = torch.concat([self.recorder.activity[0].view(l1, -1),
                                       self.recorder.activity[1].view(l2, -1)]
                                      ).cpu()
        labels = torch.concat([torch.zeros(l1), torch.ones(l2)])

        self.recorder.clear_record()
        return all_embeddings, labels

    def discrimination_performance(self, center, delta):
        data, labels = self.get_activity(center, delta)

        mcd0 = MinCovDet().fit(data[labels == 0])
        mu_robust0 = mcd0.location_
        cov_robust0 = mcd0.covariance_
        
        mcd1 = MinCovDet().fit(data[labels == 1])
        mu_robust1 = mcd1.location_
        cov_robust1 = mcd1.covariance_

        return mahalanobis(mu_robust0, mu_robust1, np.linalg.inv(0.5*cov_robust1 + 0.5*cov_robust0)) / delta

    def Fisher_info(self, angles, delta=0.03):
        """ Single point Fisher information.
            Future work: update to use multiple deltas
        """
        FIs = []
        for angle in tqdm(angles):
            dprime = self.discrimination_performance(angle, delta)
            FIs.append(dprime/delta)
        return np.array(FIs)



