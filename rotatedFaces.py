# splitting out the working models that we will focus on.

import torch
import numpy as np
import pytorch_lightning as pl

from torch import nn
from torch.utils.data import Dataset, DataLoader

from sklearn import datasets
from skimage.transform import rotate
from numpy.random import default_rng


# Models
class FaceAngleEstimator(pl.LightningModule):
    """
        General Face angle estimation.

        I'm trying to avoid too much fiddling with network parameters.
        I also want to avoid dropout if at all possible to avoid stocahsticity
    """

    def __init__(self, loss_pair, distribution_pair, encoding, **kwargs):
        super(FaceAngleEstimator, self).__init__()
        hyperparameterValues = {
            # architecture hyperparameters
            'hidden_dims': (60, 20),
            # training hyperparameters
            'lr': 1E-3,
            'batchsize': 32,
            'max_epochs': 1500,
            'dataSize': 1024,
            'gradient_clip_val': 0.5,
            'loss_pair': loss_pair,
            # stimulus generation hyperparameters
            'train_val_test_split': (300, 50, 50),
            'pixelDim': 64,
            'seed': torch.random.seed(),
            # angle distribution hyperparameters
            'distribution_pair': distribution_pair,
            'encoding': encoding
        }
        hyperparameterValues.update(kwargs)
        self.save_hyperparameters(hyperparameterValues)

        # set seeds
        torch.manual_seed(self.hparams.seed)
        np.random.seed(self.hparams.seed % (2**32-1))

        # 
        self.get_angle_samples = distribution_pair[1]
        self.face_id_split = face_id_split(self.hparams.train_val_test_split)

        self.model = nn.Sequential(
            nn.Linear(self.hparams.pixelDim**2, self.hparams.hidden_dims[0]),
            nn.CELU(),
            nn.Linear(self.hparams.hidden_dims[0], self.hparams.hidden_dims[1]),
            nn.CELU(),
            nn.Linear(self.hparams.hidden_dims[1], 2)
        )

        eps = 1E-8

        loss = loss_pair[1]
        if encoding == 'linear':
            # baseline distance measure: project onto the main diagonal, and measure
            # lower is better
            def project(embeddings):
                unit_vector = torch.ones(2, 1) / 2**0.5
                unit_vector = unit_vector.to(embeddings.device)
                return (embeddings @ unit_vector) @ unit_vector.T

            def positive_diff(x, y):
                return torch.abs(project(x)[:, 0] - project(y)[:, 0]) + eps

            self.lossFn = lambda x, y: torch.mean(loss(positive_diff(x, y)))
            self.linear_loss = lambda x, y: torch.mean(positive_diff(x, y))

        if encoding == 'circular':
            # decode based on orientation relative to the origin
            cosSim = nn.CosineSimilarity()
            relu = nn.ReLU()

            def positive_sim(x, y):
                return relu(cosSim(x, y) + 1 + eps/2) + eps / 2

            self.lossFn = lambda x, y: loss(torch.tensor(2.)) - torch.mean(loss(positive_sim(x, y)))
            self.linear_loss = lambda x, y: 2. - torch.mean(positive_sim(x, y))

    def encodeAngles(self, angles):
        if self.hparams.encoding == 'linear':
            return (angles * torch.ones(2, angles.shape[0], device=angles.device)).T
        if self.hparams.encoding == 'circular':
            return torch.stack((torch.cos(angles), torch.sin(angles)), dim=1)

    def decodeAngles(self, encodings):
        if self.hparams.encoding == 'linear':
            return (encodings  @ torch.ones(2, 1, device=encodings.device)/2).T
        if self.hparams.encoding == 'circular':
            return torch.atan2(encodings[:, 1], encodings[:, 0])

    def forward(self, images):
        """ run the network """
        nsamples = images.shape[0]
        return self.model(images.view(nsamples, -1))

    def training_step(self, batch, batchidx):
        images = batch['image']
        targets = batch['angle']

        encodedTargets = self.encodeAngles(targets)
        prediction = self.forward(images)

        loss = self.lossFn(prediction, encodedTargets)
        self.log('Train Loss', loss.item())

        return loss

    def validation_step(self, batch, batchidx):
        images = batch['image']
        targets = batch['angle']

        encodedTargets = self.encodeAngles(targets)
        prediction = self.forward(images)

        loss = self.lossFn(prediction, encodedTargets)

        linear_loss = self.linear_loss(prediction, encodedTargets)

        self.log_dict({'Val Loss': loss.item(),
                       'Linear Val Loss': linear_loss.item()})

    def test_step(self, batch, batchidx):
        images = batch['image']
        targets = batch['angle']

        encodedTargets = self.encodeAngles(targets)
        prediction = self.forward(images)

        loss = self.lossFn(prediction, encodedTargets)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    # data
    def setup(self, stage=None, sample_angles=None):
        """generate the datasets"""
        basesize = self.hparams.dataSize

        trainAngles = self.get_angle_samples(8*basesize)
        valAngles = self.get_angle_samples(2*basesize)
        testAngles = self.get_angle_samples(2*basesize)

        # generate datasets
        self.trainData = FaceDataset(trainAngles, self.face_id_split['train'])
        self.valData = FaceDataset(valAngles, self.face_id_split['val'])
        self.testData = FaceDataset(testAngles, self.face_id_split['test'])

    def train_dataloader(self):
        return DataLoader(self.trainData,
                          batch_size=self.hparams.batchsize,
                          shuffle=True,
                          num_workers=8,
                          persistent_workers=True
                          )

    def val_dataloader(self):
        return DataLoader(self.valData, batch_size=2*self.hparams.dataSize,
                          num_workers=2,
                          persistent_workers=True
                          )

    def test_dataloader(self):
        return DataLoader(self.testData, batch_size=2*self.hparams.dataSize)

    def teardown(self, stage):
        """ clearing the memory after it is used """
        if stage == "fit":
            del self.trainData
            del self.valData
            del self.testData


# Datasets
class FaceDataset(Dataset):
    """A dataset of rotated faces"""
    def __init__(self, angles, image_indices):
        super(FaceDataset, self).__init__()
        data = datasets.fetch_olivetti_faces()
        images = data['images']

        # repeat images until we have enough for all of the angles
        num_repeats = int(np.ceil(len(angles) / len(image_indices)))
        all_inds = np.tile(image_indices, num_repeats)
        image_inds = all_inds[0:len(angles)]

        # process
        images = images[image_inds]
        images = rotate_images(images, np.rad2deg(angles))
        images = mask_images(images)

        self.angles = torch.as_tensor(angles, dtype=torch.float)
        self.images = torch.as_tensor(images, dtype=torch.float).contiguous()

    def __len__(self):
        return len(self.angles)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        batch_images = self.images[idx, :, :]
        batch_angles = self.angles[idx]

        return {'image': batch_images, 'angle': batch_angles}


# utilities
def face_id_split(numbers=(300, 50, 50)):
    """ Consistant split of the faces in the dataset
        In this case, I'm using all face identities in each split
    """
    ntrain, nval, ntest = numbers

    rng = default_rng(seed=787)
    inds = rng.permutation(400)

    test_inds = inds[0:ntest]
    train_inds = inds[ntest: ntest+ntrain]
    val_inds = inds[ntest+ntrain:]

    return {'train': train_inds, 'val': val_inds, 'test': test_inds}


def mask_images(images):
    """ Add a circular mask to prevent  """
    X_mask, Y_mask = np.meshgrid(np.arange(64) - 31.5, np.arange(64) - 31.5)

    for image in images:
        image[X_mask**2 + Y_mask**2 > 31.5**2] = 0.5

    return images


def rotate_images(images, angles):
    """ rotate the images """
    for i in range(images.shape[0]):
        images[i] = rotate(images[i], angles[i])

    return images
