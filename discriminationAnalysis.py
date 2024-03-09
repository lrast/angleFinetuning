# measurement of discrimination performance by the networks

import torch
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from datageneration.stimulusGeneration import generateGrating

from scipy.signal import savgol_filter


# Smoothed derivatives
def Fisher_smooth_fits(model, theta_start, theta_end,
                       N_mean=10000, N_cov=1000, Samp_cov=1000):
    """ Fisher information: use smoothing to get the derivative """
    mean_angles = np.linspace(theta_start, theta_end, N_mean)
    neural_activity = generate_samples(model, mean_angles)

    derivs0 = savgol_filter(neural_activity[:, 0],
                            window_length=700, polyorder=2, deriv=1,
                            mode='wrap', delta=mean_angles[1]-mean_angles[0])
    derivs1 = savgol_filter(neural_activity[:, 1],
                            window_length=700, polyorder=2, deriv=1,
                            mode='wrap', delta=mean_angles[1]-mean_angles[0])

    derivs = np.array([derivs0, derivs1])

    cov_angles = np.linspace(theta_start, theta_end, N_cov)

    deriv_cov_ratio = N_mean // N_cov

    FI = []
    for i, angle in enumerate(cov_angles):
        noisy_results = generate_samples(model,  Samp_cov*[angle])
        invcov = np.linalg.inv(np.cov(noisy_results.T))

        FI.append(derivs[:, deriv_cov_ratio*i] @ invcov @ derivs[:, deriv_cov_ratio*i])

    return np.array(FI)


# linear decoding
def Fisher_discrimination(model, theta, dtheta=0.05, N_samples=1000):
    """ Measures linear Fisher information by training a linear decoder """
    # test me 

    def sample_curve(mid_point, dtheta, N, linear=True):
        """ generate samples from the model """
        if linear:
            to_sample = np.linspace(mid_point-dtheta, mid_point+dtheta, N)
        else:
            to_sample = N//2*[mid_point - dtheta] + N//2*[mid_point + dtheta]

        samples = generate_samples(model, to_sample)

        return pd.DataFrame({'stimulus': to_sample, 'r1': samples[:, 0],
                             'r2': samples[:, 1]})

    train_resp = sample_curve(theta, 2*dtheta, N_samples, linear=True)
    lin = smf.ols('stimulus ~ r1 + r2', data=train_resp)
    lin = lin.fit()

    test_resp = sample_curve(theta, dtheta, N_samples, linear=False)

    o1 = lin.predict(test_resp[0:N_samples//2])
    o2 = lin.predict(test_resp[N_samples//2:])
    
    return ((o1.mean() - o2.mean())/0.1)**2 / (0.5*(o1.var() + o2.var()))


# direct sampling
def Fisher_sampling(model, theta, dtheta=0.05, N_samples=1000, biasCorrect=True):
    """ Measure the linear Fisher by directly mean derivatives and covariances"""
    # test me

    centered = generate_samples(model, N_samples*[theta], N_samples)
    bigger = generate_samples(model, N_samples*[theta+dtheta], N_samples)
    smaller = generate_samples(model,  N_samples*[theta-dtheta], N_samples)

    covMat = np.cov(centered.T)
    tuningDeriv = (bigger.mean(0) - smaller.mean(0)) / (2*dtheta)

    fisher = tuningDeriv @ np.linalg.inv(covMat) @ tuningDeriv

    if not biasCorrect:
        return fisher

    N_neurons = tuningDeriv.shape[0]
    bcFisher = fisher * (2*N_samples - N_neurons - 3)/(2*N_samples - 3) - \
                    2*N_neurons / (N_samples*0.1**2)

    return bcFisher


def generate_samples(model, thetas, pixelDim=101, shotNoise=0.8, noiseVar=20):
    """ generate samples from the model """
    samples = model.forward(generateGrating(thetas, pixelDim=pixelDim,
                                            shotNoise=shotNoise, noiseVar=noiseVar)
                            ).detach().numpy()
    return samples


########################### Direct decoding #########################
#####

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
