# implementation of the adaptation, Fisher information fit loop
import torch
import numpy as np
import pandas as pd

from adaptableModel import AngleDistribution, AdaptableEstimator
from discriminationAnalysis import Fisher_smooth_fits

from trainers import runEarlyStoppingTraining

fit_fisher = lambda model: Fisher_smooth_fits(model, 0., np.pi, 
                                              N_mean=10000, N_cov=500, Samp_cov=500)

run_training = lambda model, directory: runEarlyStoppingTraining(model, directory,
                                                                 project='angleFineTuning')


def adapt_fit_loop(checkpoint, directory,
                   initial_dist=np.ones(500),
                   run_training=run_training, fit_fisher=fit_fisher,
                   criterion=0.1, replicates=5, max_iter=100):
    """ Running the adaptation, analysis loop """

    stimulus_dist = AngleDistribution(initial_dist, [0, np.pi])

    converged = False
    count = 0

    rows = [{'measurement': 'probability', 'iteration': count, 'data': initial_dist}]

    while not converged:
        print('#############################', count, '#############################')

        fisher_curves = []
        for replicate in range(replicates):
            model = AdaptableEstimator.load_from_checkpoint(checkpoint,
                                                            angle_dist=stimulus_dist,
                                                            max_epochs=1000,
                                                            seed=torch.random.seed()
                                                            )

            run_training(model, directory+f'/iter{count}')
            fisher_info = fit_fisher(model)

            fisher_curves.append(fisher_info)
            row = {'measurement': 'FI', 'iteration': count, 'data': fisher_info}
            rows.append(row)

        # termination criterion: uniform Fisher information
        smoothed_mean_fisher = moving_average(np.mean(fisher_curves, axis=0))

        if smoothed_mean_fisher.var() / smoothed_mean_fisher.mean() < criterion:
            converged = True

        count += 1
        if count > max_iter:
            converged = True

        if not converged:
            # make new stimulus distribution
            new_values = stimulus_dist.values / smoothed_mean_fisher**0.5

            rows.append({'measurement': 'probability', 'iteration': count+1, 'data': new_values})
            stimulus_dist = AngleDistribution(new_values, [0, np.pi])

    pd.DataFrame(rows).to_pickle(directory + '/iterate_data.pickle')


def moving_average(fisher_curve, width=31):
    """ Convolutional smoothing  """
    conv_filter = np.ones(width) / width

    edge_width = int(width/2)
    N = len(fisher_curve)
    wrapped_fisher = np.concatenate([fisher_curve[N-edge_width:N], fisher_curve,
                                     fisher_curve[0:edge_width]])

    return np.convolve(wrapped_fisher, conv_filter, mode='valid')
