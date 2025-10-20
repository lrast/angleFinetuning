# Linear discriminator on activity data
import torch

from sklearn.svm import SVC
from data.rotated_faces import ConsistentRotationDataset


class ActivityRecord():
    """Recorder to capture internal model activity"""
    def __init__(self, module):
        self.activity = {}
        module.register_forward_hook(self.record_activity)

        self.activity = {}
        self.key = 'none'

    def set_key(self, key):
        self.key = key

    def record_activity(self, module, input, output):
        key = self.key
        output = output.clone().detach().cpu()

        if key not in self.activity:
            self.activity[key] = output
        else: 
            self.activity[key] = torch.concat([self.activity[key], output])

    def clear_record(self):
        self.activity = {}


def linear_discriminability(model, module, left, delta):
    """ performance of linear discriminator on internal activity data"""
    ds1 = ConsistentRotationDataset(left-delta/2, split='valid')
    ds2 = ConsistentRotationDataset(left+delta/2, split='valid')

    dl1 = torch.utils.data.DataLoader(ds1, batch_size=256, shuffle=False)
    dl2 = torch.utils.data.DataLoader(ds2, batch_size=256, shuffle=False)

    recorder = ActivityRecord(module)

    recorder.set_key(0)
    for batch in iter(dl1):
        model.forward(batch[0].to(model.device))

    recorder.set_key(1)
    for batch in iter(dl2):
        model.forward(batch[0].to(model.device))

    l1 = len(recorder.activity[0])
    l2 = len(recorder.activity[1])
    all_embeddings = torch.concat([recorder.activity[0].view(l1, -1),
                                   recorder.activity[1].view(l2, -1)])
    labels = torch.concat([torch.zeros(l1), torch.ones(l2)])

    svm = SVC(kernel='linear', C=1.)
    svm.fit(all_embeddings, labels)

    return (svm.predict(all_embeddings.view(l1 + l2, -1)) != labels.numpy()).sum()






