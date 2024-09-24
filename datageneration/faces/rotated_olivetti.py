import numpy as np
import torch

from sklearn import datasets
from skimage.transform import rotate

from torch.utils.data import Dataset



def mask_images(images):
    """ Add a circular mask to prevent  """
    X_mask, Y_mask = np.meshgrid( np.arange(64) -31.5, np.arange(64) -31.5)

    for image in images:
        image[X_mask**2 + Y_mask**2 > 31.5**2] = 0.5

    return images


def rotate_images(images, angles):
    """ rotate the images """
    for i in range(images.shape[0]):
        images[i] = rotate(images[i], angles[i])

    return images


def train_test_split(numbers=(256, 64, 80)):
    """ Consistant train-test split 
        In this case, I'm using all face identities in each split
    """
    ntrain, nval, ntest = numbers

    np.random.seed(787)
    inds = np.random.permutation(400)
    test_inds = inds[0:ntest]
    train_inds = inds[ntest: ntest+ntrain]
    val_inds = inds[ntest+ntrain:]

    return train_inds, val_inds, test_inds


class FaceDataset(Dataset):
    """A dataset of rotated faces"""
    def __init__(self, angles, split='train', **kwargs):
        super(FaceDataset, self).__init__()
        data = datasets.fetch_olivetti_faces()
        images = data['images']

        if numbers in kwargs:
            tri, vali, testi = train_test_split(numbers)
        else:
            tri, vali, testi = train_test_split()

        if split == 'train':
            image_inds = tri
        if split == 'val' or split == 'validation':
            image_inds = vali
        if split == 'test':
            image_inds = testi

        # repeat images until we have enough for all of the angles
        num_repeats = int(np.ceil( len(angles) / len(image_inds) ))
        all_inds = np.tile(image_inds, num_repeats)
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




