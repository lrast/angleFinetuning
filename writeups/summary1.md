# Interim summary 07/23/24

I've been away from this project for a little while, but I need to return to it. The primary objective here is to get reasonable results that can be incorporated into my job talk, and maybe into a draft paper. First, I want to summarize the state of play as I remember it

1. Technical issues: 
    These issues mostly have to do with the difficulty of the fitting step:
    - The Fisher information is itself difficult to fit
        * perhaps an alternative fitting approach, like density could be used
    - The noisiness is amplified by the iterative approach because there is not enough pressure for reverting toward the mean, at least at the level of the smallest fluctuations
        * fitting parametric distributions would dampen the fluctuations

2. Results from examples:
    - Most notably, there is a significant difference between re-initializing the optimizer and keeping the same optimizer under distribution shifts. 
    - re-initialized optimizers are _worse_ at adapting to different input distributions, in the sense that Fisher information values are correlated within, but not between individual initialization runs.
    - that said, they are also less correlated overall, so it's difficult to know what to think.

Right now, I don't really see the interesting factor. The ideas that I thought would be interesting to pursue were
1. Representation (objectives and constraints) at different parts of the network
2. Representation (objectives and constraints) at different points in training
3. Maybe: representation training under different constraints

Really, all of these _still_ need the iterative process to be working! That seems like the key important factor.

Simple direction:
Do different loss functions result in different ouput distributions?
One question that arose immediately from the early results was a question of the magnitude of the encoding vector. Does this resemble certainty or some other Bayesian quantity? Is this impacted by the loss function that we use?
