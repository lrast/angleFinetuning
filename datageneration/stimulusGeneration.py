# generating angled grating stimuli
import torch
import numpy as np

from numpy.random import binomial, normal
from torch.utils.data import Dataset


def generateGrating(thetas, frequency=5, pixelDim=201, shotNoise=0., noiseVar=0.):
    """Generated angled grating with specified frequency and orientation"""
    thetas = torch.as_tensor(thetas, dtype=torch.float32)

    squeezeOutput = False
    if len(thetas.shape) == 0:  # single theta request
        thetas = torch.tensor([thetas.item()])
        squeezeOutput = True

    xs = torch.linspace(-torch.pi, torch.pi, pixelDim)
    ys = torch.linspace(-torch.pi, torch.pi, pixelDim)

    X, Y = torch.meshgrid(xs, ys, indexing='ij')

    X = X.repeat((len(thetas), 1, 1))
    Y = Y.repeat((len(thetas), 1, 1))

    Z = torch.cos(frequency * (
                    Y*torch.cos(thetas)[:, None, None] + 
                    X*torch.sin(thetas)[:, None, None]
                    )
                  )

    # add noise to the generated gratings
    noiseLocations = binomial(1, shotNoise,
                              size=(len(thetas), pixelDim, pixelDim))
    noiseMagnitude = normal(scale=noiseVar**0.5,
                            size=(len(thetas), pixelDim, pixelDim))

    Z = torch.clamp(Z + torch.tensor(
                         noiseLocations * noiseMagnitude, dtype=torch.float32
                        ), min=-1., max=1.)

    r2 = X**2 + Y**2
    Z[r2 >= 6] = torch.zeros(Z[r2 >= 6].shape)

    if squeezeOutput:
        return Z[0]

    return Z


class GratingDataset(Dataset):
    """A dataset of gratings"""
    def __init__(self, angles, **gratingKwargs):
        super(GratingDataset, self).__init__()
        self.angles = torch.as_tensor(angles, dtype=torch.float)
        self.images = generateGrating(angles, **gratingKwargs)

    def __len__(self):
        return len(self.angles)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        batch_images = self.images[idx, :, :]
        batch_angles = self.angles[idx]

        return {'image': batch_images, 'angle': batch_angles}


def searchForAngle(targetGrating, endpoints, targetResolution, makeGrating):
    """Positive control: 
        repeatedly search for the angle with maximum overlapping grating
    """
    angles = np.linspace(endpoints[0], endpoints[1], 20)
    
    overlaps = []
    for angle in angles:
        test = makeGrating(angle)
        overlaps.append(findOverlap(targetGrating, test))
    
    maxInd = np.argmax(np.array(overlaps))
    
    if endpoints[1] - endpoints[0] < targetResolution:
        return angles[maxInd]
    
    nextMin = angles[max(maxInd-1, 0)]
    nextMax = angles[min(maxInd+1, len(angles)-1)]
    
    return searchForAngle(targetGrating, 
                          [nextMin, nextMax], targetResolution, makeGrating)


def findOverlap(grating1, grating2):
    return (grating1*grating2).sum()
