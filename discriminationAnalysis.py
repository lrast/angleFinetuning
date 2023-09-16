# measurement of discrimination performance by the networks

import numpy as np
import torch

from datageneration.stimulusGeneration import generateGrating


def sensitivityIndex(model, angle, samples=1000, dtheta=0.01,
                     pixelDim=101, shotNoise=0.8, noiseVar=20):
    """Evaluate sensitivity to changes in stimulus"""
    stimuli = generateGrating(samples*[angle], pixelDim=pixelDim,
                              shotNoise=shotNoise, noiseVar=noiseVar)
    baseAngles = rotateAndDecode(angle, model.forward(stimuli))

    stimuli = generateGrating(samples*[angle+dtheta], pixelDim=pixelDim,
                              shotNoise=shotNoise, noiseVar=noiseVar)
    perturbedAngles = rotateAndDecode(angle, model.forward(stimuli))

    std = ((baseAngles.var() + perturbedAngles.var())/2)**0.5
    delta = torch.abs(baseAngles.mean() - perturbedAngles.mean())

    return (delta / std).item()


def modelBias(model, angle, samples=1000,
              pixelDim=101, shotNoise=0.8, noiseVar=20):
    stimuli = generateGrating(samples*[angle], pixelDim=pixelDim,
                              shotNoise=shotNoise, noiseVar=noiseVar)
    angles = rotateAndDecode(angle, model.forward(stimuli))

    return (angles.mean() - angle).item()


def rotateAndDecode(angle, encoding):
    """
        Rotate the encodings to be centered at zero degrees, then decode
        This is useful for distributions that are tightly spaced.
        Note that with the encoding at hand, this is a rotation
        of positive double the angle
    """
    rotMat = torch.tensor([[np.cos(2*angle), -np.sin(2*angle)],
                           [np.sin(2*angle), np.cos(2*angle)]],
                          dtype=encoding.dtype)
    rotated = encoding @ rotMat
    return torch.atan2(rotated[:, 1], rotated[:, 0]) / 2 + angle
