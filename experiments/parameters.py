# parameters that stay relatively constant between experiments
import numpy as np
import torch

# parameter sets for von mises distributions
uniformConfig = {
             'loc_tr': np.pi/2,
             'kappa_tr': 1E-8,
             'loc_val': np.pi/2,
             'kappa_val': 1E-8,
             'loc_test': np.pi/2,
              'kappa_test': 1E-8
             }

concentratedConfig_1 = {
         'loc_tr': np.pi/2,
         'kappa_tr': 3.,
         'loc_val': np.pi/2,
         'kappa_val': 3.,
         'loc_test': np.pi/2,
         'kappa_test': 3.
         }

concentratedConfig_2 = {
         'loc_tr': np.pi/2,
         'kappa_tr': 1.,
         'loc_val': np.pi/2,
         'kappa_val': 1.,
         'loc_test': np.pi/2,
         'kappa_test': 1.
         }

# Loss functions over model similarity outputs: decreasing in similarity
lossFns = {
    'linear': lambda d: 2. - torch.mean(d),
    'log': lambda d: np.log(2.) - torch.mean(torch.log(d)),
    'sqrt': lambda d: np.sqrt(2.) - torch.mean(torch.sqrt(d))
}

# Loss functions over model distance outputs: increasing in distance
lossFns_decreasing = {
    'linear': lambda d: torch.mean(d),
    'square': lambda d: torch.mean(d**2),
    'sqrt': lambda d: torch.mean(torch.sqrt(d))
}
