# measurement of discrimination performance by the networks

import torch
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from datageneration.stimulusGeneration import generateGrating


# linear decoding
def Fisher_discrimination(model, theta, dtheta=0.05, N_samples=1000,
                          pixelDim=101, shotNoise=0.8, noiseVar=20
                          ):
    """ Measures linear Fisher information by training a linear decoder """

    def generate_samples(mid_point, dtheta, N, linear=True):
        """ generate samples from the model """
        if linear:
            to_sample = np.linspace(mid_point-dtheta, mid_point+dtheta, N)
        else:
            to_sample = N//2*[mid_point - dtheta] + N//2*[mid_point + dtheta]

        samples = model.forward(generateGrating(to_sample, pixelDim=pixelDim,
                                                shotNoise=shotNoise, noiseVar=noiseVar)
                                ).detach().numpy()

        return pd.DataFrame({'stimulus': to_sample, 'r1': samples[:, 0],
                             'r2': samples[:, 1]})

    train_resp = generate_samples(theta, 2*dtheta, N_samples, linear=True)
    lin = smf.ols('stimulus ~ r1 + r2', data=train_resp)
    lin = lin.fit()

    test_resp = generate_samples(theta, dtheta, N_samples, linear=False)

    o1 = lin.predict(test_resp[0:N_samples//2])
    o2 = lin.predict(test_resp[N_samples//2:])
    
    return ((o1.mean() - o2.mean())/0.1)**2 / (0.5*(o1.var() + o2.var()))


# direct sampling
def Fisher_sampling(model, theta, dtheta=0.05, N_samples=1000, biasCorrect=True,
                    pixelDim=101, shotNoise=0.8, noiseVar=20
                    ):
    """ Measure the linear Fisher by directly mean derivatives and covariances"""
    def make_samples(angle, N):
        return model.forward(generateGrating(N*[angle], pixelDim=pixelDim,
                                             shotNoise=shotNoise, noiseVar=noiseVar)
                             ).detach().numpy()

    centered = make_samples(theta, N_samples)
    bigger = make_samples(theta, N_samples)
    smaller = make_samples(theta, N_samples)

    covMat = np.cov(centered.T)
    tuningDeriv = (bigger.mean(0) - smaller.mean(0)) / (2*dtheta)

    fisher = tuningDeriv @ np.linalg.inv(covMat) @ tuningDeriv

    if not biasCorrect:
        return fisher

    N_neurons = tuningDeriv.shape[0]
    bcFisher = fisher * (2*N_samples - N_neurons - 3)/(2*N_samples - 3) - \
                    2*N_neurons / (N_samples*0.1**2)

    return bcFisher


# Direct decoding
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
