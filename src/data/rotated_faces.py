# rotated faces dataset

import torch
import numpy as np

from scipy.stats import vonmises
from torch.utils.data import Dataset
from torchvision.datasets import CelebA

import torchvision.transforms as T
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import rotate, resize


class RotatedFacesDataset(Dataset):
    def __init__(self, loc=0, kappa=1, split='train'):
        """
        Args:
            split: split of the data to load 
        """
        self.transform = T.Compose([
            T.CenterCrop((178, 178)),
            T.Resize((210, 210), interpolation=InterpolationMode.BILINEAR),
            T.ToTensor(),
        ])

        self.base_dataset = CelebA(
            root='~/Datasets',
            split=split,
            download=True,
            transform=self.transform
        )

        self.angle_dist = vonmises(loc=loc, kappa=kappa)

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        raw_image = self.base_dataset[idx][0]
        angle = self.angle_dist.rvs() 

        image = resize(mask_image(
                        rotate(raw_image, angle * 180 / torch.pi,
                               interpolation=InterpolationMode.BILINEAR
                               )), (200, 200))
        angle_vector = torch.tensor([np.cos(angle), np.sin(angle)]).to(torch.float32)

        return image, angle_vector


class SingleFaceDataset(RotatedFacesDataset):
    """ SingleFaceDataset: a version of a RotatedFacesDataset that returns a 
    single face.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        base_dataset = CelebA(
            root='~/Datasets',
            split='train',
            download=True,
            transform=self.transform
        )
        self.base_dataset = OnlyFirstWrapper(base_dataset)

    def __len__(self):
        return 500


class OnlyFirstWrapper():
    """ Dataset that consists only of the first element in the base dataset """
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, idx):
        return self.dataset[0]


class ConsistentRotationDataset(RotatedFacesDataset):
    """ SingleFaceDataset: a version of a RotatedFacesDataset that returns a 
    single face.
    """
    def __init__(self, angle, **kwargs):
        super().__init__(**kwargs)

        self.angle_dist = DeltaSampler(angle)


class DeltaSampler():
    """ (Not) random sampler that consistently returns one value"""
    def __init__(self, angle):
        self.angle = angle

    def rvs(self):
        return self.angle


def mask_image(image):
    """ Add a circular mask to prevent  """
    X_mask, Y_mask = np.meshgrid(np.arange(210) - 104.5, np.arange(210) - 104.5)

    image[:, X_mask**2 + Y_mask**2 > 104.5**2] = 0.5

    return image


if __name__ == '__main__':
    # debugging mode
    batch = next(iter(
        torch.utils.data.DataLoader(RotatedFacesDataset(), batch_size=8, num_workers=2)
        ))
    print(batch[0].dtype, batch[1].dtype)
