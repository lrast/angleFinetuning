# measurement of discrimination performance by the networks

import torch
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from datageneration.stimulusGeneration import generateGrating
from datageneration.faces.rotated_olivetti import FaceDataset

from scipy.signal import savgol_filter


# adaptive sampling approaches for rough Fisher information functions

def interval_Fisher(model, i0, i1, input_samples=20, target_sterr_percent=0.02, max_samples=100, **FIkwargs):
    """
        The Fisher information of learned models seems to vary quite a lot
        with even small changes in where we evaluate it.

        Instead of point evaluations, sample Fisher information through
        an interval and use averages w/ bootstrap confidence estimates
    """

    delta = (i1 - i0)
    sample_points = i0 + delta * np.random.rand(input_samples)
    FIs = Fisher_derivatives_faces(model, sample_points, **FIkwargs)

    standard_error = FIs.std() / FIs.shape[0]**0.5
    target_sterr = target_sterr_percent * FIs.mean()

    print(input_samples, (i0, i1), standard_error)

    if standard_error < target_sterr:
        return (i0, i1), FIs.mean(), standard_error
    else:
        # how many samples do we need to achieve our target variance
        fudge = 1. # adjustment for
        samples_needed = input_samples * (standard_error / target_sterr)**2
        samples_needed = int(samples_needed) + 10
        print(samples_needed)
        if samples_needed > max_samples:
            # we won't hit our target by increasing the number of samples:
            # split the interval, and increase the number of samples in the individual evaluations
            results1 = interval_Fisher(model, i0, i0+delta/2, input_samples=input_samples,
                                       target_sterr_percent=target_sterr_percent, max_samples=max_samples, **FIkwargs)
            results2 = interval_Fisher(model, i0+delta/2, i1, input_samples=input_samples,
                                       target_sterr_percent=target_sterr_percent, max_samples=max_samples, **FIkwargs)

            return ['split', results1, results2]
        else:
            # increase the number of samples
            return interval_Fisher(model, i0, i1, input_samples=samples_needed, target_sterr_percent=target_sterr_percent,
                                   max_samples=max_samples, **FIkwargs)


def Full_Fisher_direct(model, thetas, **passThrough):
    """
        Assemble interval Fisher information data
        indexed to the front of the iterval
    """
    thetas_out = []
    FIs_out = []

    def parse_results(results):
        """ parse the results tree"""
        if results[0] == 'split':
            parse_results(results[1])
            parse_results(results[2])
        else:
            # use the interval midpoint
            interval, FI, standard_error = results
            thetas_out.append((interval[1] + interval[0])/2)
            FIs_out.append(FI)

    for i in range(len(thetas)-1):
        results = interval_Fisher(model, thetas[i], thetas[i+1], **passThrough)
        parse_results(results)

    ind_sort = np.argsort(thetas_out)
    return np.array(thetas_out)[ind_sort], np.array(FIs_out)[ind_sort]


def Full_Fisher_max_delta(model, thetas, FindFisherInfo, min_diff=10., min_diff_fold=0.05, min_interval=1E-3):
    """ Adjust the sampling frequency to capture variable locations
        by adding points until the changes between points are below a maximum value.

        This is an attempt to strike a happy medium between single evaluations which
        are very location dependent, and interval averages, which are expensive.
        In that interest, we allow the either small absolute changes or small fold changes in Fisher info.

        min_diff: minimum Fisher information difference allowed
        min_diff_fold: minimum fractional Fisher information difference allowed
        min_interval: smallest interval that we test before concluding that there is 
                      a discontinuity in the Fisher information
    """

    def get_coverage(thetas):
        """ recursively cover intervals until the differences are small enough  """

        thetas_out = []
        FIs_out = []

        FIs = FindFisherInfo(model, thetas)

        #print(thetas, FIs)
        # where do we have differences that are too large?
        diffs = np.abs(np.diff(FIs))
        fold_difference_ratio = diffs / (min_diff_fold*(FIs[:-1] + FIs[1:])/2)
        absolute_difference_ratio = diffs / min_diff

        difference_ratio = np.minimum(fold_difference_ratio, absolute_difference_ratio)

        n = thetas.shape[0]
        for i in range(n-1):
            thetas_out.append(thetas[i])
            FIs_out.append(FIs[i])

            if thetas[i+1] - thetas[i] <= min_interval:
                # conclude that there is a discontinuity here
                continue

            elif difference_ratio[i] > 1:
                # there is room for improvement
                n_new = int(difference_ratio[i]) + 2 # minimum of 3 new points
                thetas_new = np.linspace(thetas[i], thetas[i+1], n_new)

                thetas_subinterval, FIs_subinterval = get_coverage(thetas_new)

                # don't add the first or last points
                thetas_out.extend(thetas_subinterval[1:-1])
                FIs_out.extend(FIs_subinterval[1:-1])

        # add on the last point
        thetas_out.append(thetas[n-1])
        FIs_out.append(FIs[n-1])

        return thetas_out, FIs_out

    thetas_out, FIs = get_coverage(thetas)
    return np.array(thetas_out), np.array(FIs)


# taking advantage of neural network derivatives
def Fisher_derivatives_faces(model, thetas, num_samples=1000, image_delta=0.05):
    """ Direct evaluation of the Fisher information by taking derivatives of the neural
        networks.

        Written specifically for the face dataset.

        In notebook 2.2, I explore how empirically this approach is substantially 
        more statistically efficient that direct evaluation.
    """
    def point_Fisher(theta):
        """ This could definitely be sped up by vectorization in a variety of dimensions """
        I0 = FaceDataset(torch.zeros(80) + theta, split='test').images
        I0.requires_grad = True

        outputs = model.forward(I0.to(model.device).repeat((num_samples, 1, 1))).cpu()
        cov = outputs.T.cov()

        mean_grad0 = torch.autograd.grad(outputs.mean(0)[0], I0, retain_graph=True)[0]
        mean_grad1 = torch.autograd.grad(outputs.mean(0)[1], I0, retain_graph=True)[0]

        cov_grad00 = torch.autograd.grad(cov[0,0], I0, retain_graph=True)[0]
        cov_grad01 = torch.autograd.grad(cov[0,1], I0, retain_graph=True)[0]
        cov_grad10 = torch.autograd.grad(cov[1,0], I0, retain_graph=True)[0]
        cov_grad11 = torch.autograd.grad(cov[1,1], I0)[0]

        cov = cov.detach()

        plus_I0 = FaceDataset(torch.zeros(80) + theta + image_delta/2, split='test').images
        minus_I0 = FaceDataset(torch.zeros(80) + theta - image_delta/2, split='test').images
        image_deriv = (plus_I0 - minus_I0) / image_delta 

        mean_deriv0 = (mean_grad0 * image_deriv).sum().item()
        mean_deriv1 = (mean_grad1 * image_deriv).sum().item()
        douts = torch.tensor([mean_deriv0, mean_deriv1])

        cov_deriv00 = (cov_grad00 * image_deriv).sum().item()
        cov_deriv01 = (cov_grad10 * image_deriv).sum().item()
        cov_deriv10 = (cov_grad10 * image_deriv).sum().item()
        cov_deriv11 = (cov_grad11 * image_deriv).sum().item()
        dCov = torch.tensor([[cov_deriv00, cov_deriv01], [cov_deriv10, cov_deriv11]])

        invcov = torch.linalg.inv(cov)

        return douts @ invcov @ douts + 0.5 * torch.trace(invcov @ dCov @ invcov @ dCov)

    FIs = np.zeros(len(thetas))
    for i, theta in enumerate(thetas):
        FIs[i] = point_Fisher(theta)

    torch.mps.empty_cache()
    return FIs


def Fisher_derivatives_faces_mean(model, thetas, num_samples=1000, image_delta=0.05):
    """ The mean term in the Fisher information, as evaluated by derivatives of the neural network
    """
    def point_Fisher(theta):
        """ This could definitely be sped up by vectorization in a variety of dimensions """
        I0 = FaceDataset(torch.zeros(80) + theta, split='test').images.contiguous()
        I0.requires_grad = True

        outputs = model.forward(I0.to(model.device).repeat((num_samples, 1, 1))).cpu()
        model_grad0 = torch.autograd.grad(outputs.mean(0)[0], I0, retain_graph=True)[0]
        model_grad1 = torch.autograd.grad(outputs.mean(0)[1], I0)[0]

        plus_I0 = FaceDataset(torch.zeros(80) + theta + image_delta/2, split='test').images.contiguous()
        minus_I0 = FaceDataset(torch.zeros(80) + theta - image_delta/2, split='test').images.contiguous()
        image_deriv = (plus_I0 - minus_I0) / image_delta 

        full_deriv0 = (model_grad0 * image_deriv).sum().item()
        full_deriv1 = (model_grad1 * image_deriv).sum().item()
        douts = torch.tensor([full_deriv0, full_deriv1])

        cov = outputs.T.cov().detach()
        torch.mps.empty_cache()
        return (douts @ torch.linalg.inv(cov) @ douts).item()

    FIs = np.zeros(len(thetas))
    for i, theta in enumerate(thetas):
        FIs[i] = point_Fisher(theta)

    return FIs


def Fisher_derivatives_faces_covariance(model, thetas, num_samples=1000, image_delta=0.05):
    """ The covariance term in the Fisher information, as evaluated by derivatives of the neural network
    """
    def point_Fisher(theta):
        """ This could definitely be sped up by vectorization in a variety of dimensions """
        I0 = FaceDataset(torch.zeros(80) + theta, split='test').images.contiguous()
        I0.requires_grad = True

        outputs = model.forward(I0.to(model.device).repeat((num_samples, 1, 1))).cpu()
        cov = outputs.T.cov()
        model_grad00 = torch.autograd.grad(cov[0,0], I0, retain_graph=True)[0]
        model_grad01 = torch.autograd.grad(cov[0,1], I0, retain_graph=True)[0]
        model_grad10 = torch.autograd.grad(cov[1,0], I0, retain_graph=True)[0]
        model_grad11 = torch.autograd.grad(cov[1,1], I0)[0]

        plus_I0 = FaceDataset(torch.zeros(80) + theta + image_delta/2, split='test').images.contiguous()
        minus_I0 = FaceDataset(torch.zeros(80) + theta - image_delta/2, split='test').images.contiguous()
        image_deriv = (plus_I0 - minus_I0) / image_delta 

        full_deriv00 = (model_grad00 * image_deriv).sum().item()
        full_deriv01 = (model_grad10 * image_deriv).sum().item()
        full_deriv10 = (model_grad10 * image_deriv).sum().item()
        full_deriv11 = (model_grad11 * image_deriv).sum().item()

        dCov = torch.tensor([[full_deriv00, full_deriv01], [full_deriv10, full_deriv11]])
        cov = cov.detach()
        invcov = torch.linalg.inv(cov)
        torch.mps.empty_cache()
        return 0.5 * torch.trace(invcov @ dCov @ invcov @ dCov)

    FIs = np.zeros(len(thetas))
    for i, theta in enumerate(thetas):
        FIs[i] = point_Fisher(theta)

    return FIs


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
        noisy_results = generate_samples(model, Samp_cov*[angle])
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


########################### Direct decoding ##############################

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
