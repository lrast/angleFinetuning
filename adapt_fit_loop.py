# implementation of the adaptation, Fisher information fit loop
import wandb
import numpy as np

from adaptableModel import AngleDistribution, AdaptableEstimator
from discriminationAnalysis import Fisher_smooth_fits

from trainers import runEarlyStoppingTraining

fit_fisher = lambda model: Fisher_smooth_fits(model, 0., np.pi, 
                                              N_mean=10000, N_cov=500, Samp_cov=500)

run_training = lambda model, directory: runEarlyStoppingTraining(model, directory,
                                                                 project='angleFineTuning')


def adapt_fit_loop(checkpoint, directory,
                   initialDist=np.ones(500),
                   run_training=run_training, fit_fisher=fit_fisher,
                   criterion=0.1, max_iter=100):
    """ Running the adaptation, analysis loop """

    stimulus_dist = AngleDistribution(initialDist)

    converged = False
    count = 0

    while not converged:
        print('#############################', count, '#############################')
        model = AdaptableEstimator.load_from_checkpoint(checkpoint,
                                                        angle_dist=stimulus_dist,
                                                        max_epochs=1000
                                                        )

        wandb.init(reinit=True)
        run_training(model, directory+f'iter{count}')

        fisher_curve = fit_fisher(model)

        # termination criterion: uniform Fisher information
        if fisher_curve.var() / fisher_curve.mean() < criterion:
            converged = True

        count += 1
        if count > max_iter:
            converged = True

        if not converged:
            # make new stimulus distribution
            new_values = stimulus_dist.values / fisher_curve**0.5
            stimulus_dist = AngleDistribution(new_values)
