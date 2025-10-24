# Linear discriminator on activity data
import torch
import numpy as np
import warnings

from data.rotated_faces import ConsistentRotationDataset

from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from scipy.stats import norm
from sklearn.model_selection import StratifiedKFold

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

    def get_activity(self, midpoint, delta):
        """ performance of linear discriminator on internal activity data"""

        ds1 = ConsistentRotationDataset(midpoint-delta/2, split='valid')
        ds2 = ConsistentRotationDataset(midpoint+delta/2, split='valid')

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

    def get_discrimination_performance(self, midpoint, delta=0.03):
        data, labels = self.get_activity(midpoint, delta)
        # cross validated d prime measurements
        skf = StratifiedKFold(n_splits=8)

        results = []
        for test_ind, train_ind in skf.split(data, labels):
            results.append(dprime(data[train_ind], labels[train_ind],
                                  data[test_ind], labels[test_ind]
                                  ))

        return np.mean(results), np.cov(results)

    def Fisher_info(self, angles, delta=0.03):
        """ Single point Fisher information.
            Future work: update to use multiple deltas
        """
        FIs = []
        for angle in tqdm(angles):
            dprime, var = self.get_discrimination_performance(angle, delta)
            if var / dprime > 0.1:
                warnings.warn('Large cross validation variance.')
            FIs.append(dprime/delta)
        return np.array(FIs)


def dprime(train_data, train_labels, test_data, test_labels):
    smv = SVC(kernel='linear', C=1.)
    smv.fit(train_data, train_labels)
    C = confusion_matrix(smv.predict(test_data), test_labels)
    
    dprime = norm.ppf(C[0, 0] / C.sum(0)[0]) - norm.ppf(C[0, 1] / C.sum(0)[1])
    return dprime
