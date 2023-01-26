# generating angled grating stimuli
import torch
import numpy as np

from numpy.random import binomial, normal
from torch.utils.data import Dataset


def generateGrating(theta, frequency=5, pixelDim=201, shotNoise=0., noiseVar=0.):
    """Generated angled grating with specified frequency and orientation"""
    xs = np.linspace(-np.pi, np.pi, pixelDim)
    ys = np.linspace(-np.pi, np.pi, pixelDim)

    X, Y = np.meshgrid(xs, ys)
    Z = np.cos( frequency * (Y*np.cos(theta) + X*np.sin(theta) ) )

    # add noise to the generated gratings
    noiseLocations = binomial(1, shotNoise, size=(pixelDim, pixelDim))
    noiseMagnitude = normal(scale=noiseVar**0.5, size=(pixelDim, pixelDim))

    Z = np.maximum( np.minimum( Z+noiseLocations * noiseMagnitude, 1), -1)

    r2 = X**2 + Y**2
    Z[r2 >= 6] = 0.

    return Z



class GratingDataset(object):
    """A dataset of gratings"""
    def __init__(self, angles, **gratingKwargs):
        super(GratingDataset, self).__init__()
        self.angles = torch.as_tensor( angles, dtype=torch.float)
        self.images = torch.tensor( 
            np.array(
            list( map(lambda x: generateGrating( x, **gratingKwargs), angles) ),
            ),
            dtype=torch.float )

    def __len__(self):
        return len(self.angles)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        batch_images = self.images[idx, :, :]
        batch_angles = self.angles[idx]

        return {'image': batch_images, 'angle': batch_angles}



def searchForAngle(targetGrating, endpoints, targetResolution, makeGrating):
    """Positive control: repeatedly search for the angle with maximum overlapping grating"""
    angles = np.linspace(endpoints[0], endpoints[1], 20)
    
    overlaps = []
    for angle in angles:
        test = makeGrating(angle)
        overlaps.append( findOverlap( targetGrating, test)  )
    
    maxInd = np.argmax( np.array(overlaps) )
    
    if endpoints[1] - endpoints[0] < targetResolution:
        return angles[maxInd]
    
    nextMin = angles[ max( maxInd-1, 0) ]
    nextMax = angles[ min( maxInd+1, len(angles)-1) ]
    
    return searchForAngle(targetGrating, [ nextMin, nextMax ] , targetResolution, makeGrating)


def findOverlap(grating1, grating2):
    return np.sum(grating1*grating2)

